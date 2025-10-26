#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Salon Scheduler: авто-распределение записей посетителей по календарю.

Функции:
- Хранение посетителей и ручных правок.
- Генерация годового календаря с учётом условий:
  * Рабочий день: 08:00–20:00
  * 1 сеанс = 60 минут + 10 минут пауза (итого слот 70 мин)
  * Максимум 10 сеансов в день
  * Абонемент на 6 месяцев: минимум 10 обязательных сеансов за период
  * Свободные слоты можно раздавать как бонус (опция)
- Экспорт в CSV и ICS (iCalendar).
- Команды CLI на базе argparse.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, time, timedelta
from typing import Dict, List, Optional, Tuple
import sys
import os

try:
    # dateutil облегчает прибавление месяцев, корректные границы 6-месячного периода
    from dateutil.relativedelta import relativedelta
except Exception:
    print("Для работы требуется пакет python-dateutil. Установите: pip install python-dateutil", file=sys.stderr)
    sys.exit(1)

# ------------------------------
# Константы салона и расписания
# ------------------------------
WORK_START = time(8, 0)
WORK_END   = time(20, 0)
SLOT_MINUTES = 70  # 60 мин сеанс + 10 мин пауза
SLOTS_PER_DAY = 10 # проверено: 10 * 70 мин = 700 мин = 11ч40м; окончание 19:40

TIMEZONE = "Europe/Riga"  # для ICS экспорта

# Файлы хранилища
VISITORS_FILE = "visitors.json"
MANUAL_FILE   = "manual_blocks.json"
SCHEDULE_FILE = "schedule.json"

# ------------------------------
# Модели
# ------------------------------
@dataclass
class Visitor:
    name: str
    start_date: str  # YYYY-MM-DD
    months: int = 6
    min_sessions: int = 10

    def period(self) -> Tuple[date, date]:
        start = datetime.strptime(self.start_date, "%Y-%m-%d").date()
        end = start + relativedelta(months=self.months)
        return (start, end)

@dataclass
class ManualBlock:
    date: str   # YYYY-MM-DD
    time: str   # HH:MM (начало слота)
    name: Optional[str] = None  # если указано — это ручная запись на клиента; если None — просто блокировка
    comment: Optional[str] = None

@dataclass
class Appointment:
    date: str   # YYYY-MM-DD
    time: str   # HH:MM
    name: str
    is_bonus: bool = False
    is_manual: bool = False

# ------------------------------
# Утилиты
# ------------------------------
def daterange(start: date, end: date):
    """Итерация по датам [start, end) — end не включительно."""
    d = start
    while d < end:
        yield d
        d += timedelta(days=1)

def day_slots() -> List[time]:
    """Вычисляет список стартовых времён слотов в течение рабочего дня."""
    slots = []
    cur = datetime.combine(date.today(), WORK_START)
    end_dt = datetime.combine(date.today(), WORK_END)
    delta = timedelta(minutes=SLOT_MINUTES)
    for _ in range(SLOTS_PER_DAY):
        if cur.time() > end_dt.time():
            break
        slots.append(cur.time())
        cur += delta
    return slots

SLOTS_CACHE = day_slots()

def parse_time(s: str) -> time:
    return datetime.strptime(s, "%H:%M").time()

def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()

# ------------------------------
# Хранилище
# ------------------------------
class Store:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self.visitors_path = os.path.join(self.base_dir, VISITORS_FILE)
        self.manual_path   = os.path.join(self.base_dir, MANUAL_FILE)
        self.schedule_path = os.path.join(self.base_dir, SCHEDULE_FILE)

    def load_visitors(self) -> List[Visitor]:
        if not os.path.exists(self.visitors_path):
            return []
        with open(self.visitors_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [Visitor(**x) for x in data]

    def save_visitors(self, visitors: List[Visitor]):
        with open(self.visitors_path, "w", encoding="utf-8") as f:
            json.dump([asdict(v) for v in visitors], f, ensure_ascii=False, indent=2)

    def load_manual(self) -> List[ManualBlock]:
        if not os.path.exists(self.manual_path):
            return []
        with open(self.manual_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [ManualBlock(**x) for x in data]

    def save_manual(self, blocks: List[ManualBlock]):
        with open(self.manual_path, "w", encoding="utf-8") as f:
            json.dump([asdict(b) for b in blocks], f, ensure_ascii=False, indent=2)

    def load_schedule(self) -> Dict[str, List[Appointment]]:
        if not os.path.exists(self.schedule_path):
            return {}
        with open(self.schedule_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        result: Dict[str, List[Appointment]] = {}
        for d, arr in raw.items():
            result[d] = [Appointment(**x) for x in arr]
        return result

    def save_schedule(self, schedule: Dict[str, List[Appointment]]):
        serializable = {d: [asdict(ap) for ap in arr] for d, arr in schedule.items()}
        with open(self.schedule_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)

# ------------------------------
# Планировщик
# ------------------------------
class Scheduler:
    def __init__(self, store: Store):
        self.store = store

    def build_empty_calendar(self, start: date, end: date) -> Dict[str, List[Appointment]]:
        cal: Dict[str, List[Appointment]] = {}
        for d in daterange(start, end):
            cal[d.isoformat()] = []
        return cal

    def place_manual(self, cal: Dict[str, List[Appointment]], manual: List[ManualBlock]):
        slot_set = set(t.strftime("%H:%M") for t in SLOTS_CACHE)
        for m in manual:
            if m.time not in slot_set:
                # нормализуем ко времени ближайшего слота (если случайно ввели 09:00 вместо 09:10)
                # выбираем ближайший по разнице минут
                t_input = parse_time(m.time)
                nearest = min(SLOTS_CACHE, key=lambda t: abs((datetime.combine(date.today(), t) - datetime.combine(date.today(), t_input)).total_seconds()))
                t_norm = nearest.strftime("%H:%M")
            else:
                t_norm = m.time
            ap = Appointment(date=m.date, time=t_norm, name=(m.name or "BLOCKED"), is_bonus=False, is_manual=True)
            if m.date in cal:
                # не дублировать
                if not any(x.time == t_norm for x in cal[m.date]):
                    cal[m.date].append(ap)

    def count_taken_by(self, cal: Dict[str, List[Appointment]], name: str, period: Tuple[date, date]) -> int:
        start, end = period
        total = 0
        for d in daterange(start, end):
            ds = d.isoformat()
            if ds in cal:
                total += sum(1 for ap in cal[ds] if ap.name == name)
        return total

    def day_free_slot(self, cal: Dict[str, List[Appointment]], d: date) -> Optional[str]:
        ds = d.isoformat()
        taken = set(ap.time for ap in cal.get(ds, []))
        for t in SLOTS_CACHE:
            st = t.strftime("%H:%M")
            if st not in taken:
                return st
        return None

    def schedule_minimums(self, cal: Dict[str, List[Appointment]], visitors: List[Visitor], start: date, end: date):
        # Ставим обязательные 10 сеансов каждому
        # Сначала отсортируем посетителей по началу периода (раньше — раньше планируем)
        visitors_sorted = sorted(visitors, key=lambda v: parse_date(v.start_date))
        for v in visitors_sorted:
            p_start, p_end = v.period()
            # ограничим период общими рамками планируемого календаря
            p_start = max(p_start, start)
            p_end   = min(p_end, end)
            if p_start >= p_end:
                continue
            already = self.count_taken_by(cal, v.name, (p_start, p_end))
            need = max(0, v.min_sessions - already)
            if need == 0:
                continue

            days_total = (p_end - p_start).days
            # равномерное распределение
            step = max(1, days_total // need)  # шаг в днях между целевыми датами
            target = p_start

            placed = 0
            attempts = 0
            while placed < need and attempts < need * days_total + 1000:
                # если цель вышла за период, вернём в конец периода
                if target >= p_end:
                    target = p_end - timedelta(days=1)

                # ищем ближайший свободный день начиная с target и двигаясь вперёд, затем назад
                found_date = None
                # вперёд
                d = target
                while d < p_end:
                    slot = self.day_free_slot(cal, d)
                    if slot:
                        found_date = d
                        break
                    d += timedelta(days=1)
                if found_date is None:
                    # назад
                    d = target - timedelta(days=1)
                    while d >= p_start:
                        slot = self.day_free_slot(cal, d)
                        if slot:
                            found_date = d
                            break
                        d -= timedelta(days=1)

                if found_date is None:
                    # нет свободных слотов в периоде — пропускаем (уведомление)
                    print(f"[ВНИМАНИЕ] Не удалось поставить обязательный сеанс для {v.name} в период {p_start}..{p_end}", file=sys.stderr)
                    break

                slot_time = self.day_free_slot(cal, found_date)
                if slot_time:
                    cal[found_date.isoformat()].append(Appointment(
                        date=found_date.isoformat(),
                        time=slot_time,
                        name=v.name,
                        is_bonus=False,
                        is_manual=False
                    ))
                    placed += 1
                    # следующая целевая дата
                    target = found_date + timedelta(days=step)
                attempts += 1

    def schedule_bonuses(self, cal: Dict[str, List[Appointment]], visitors: List[Visitor], start: date, end: date, max_bonus_per_person: Optional[int] = None):
        # Заполняем оставшиеся свободные слоты равномерно между посетителями, если они ещё в своём 6-месячном периоде
        # max_bonus_per_person можно ограничить, если нужно (по умолчанию не ограничиваем)
        # Индекс по посетителям для "карусели"
        bonus_count: Dict[str, int] = {}
        idx = 0
        active_visitors = visitors[:]  # копия
        # предварительная сортировка чтобы распределение было детерминированным
        active_visitors.sort(key=lambda v: (parse_date(v.start_date), v.name))

        for d in daterange(start, end):
            ds = d.isoformat()
            # перебираем свободные слоты дня
            while True:
                slot_time = self.day_free_slot(cal, d)
                if not slot_time:
                    break
                # ищем следующего посетителя, чей период включает d
                looped = 0
                assigned = False
                while looped < len(active_visitors):
                    v = active_visitors[idx % len(active_visitors)]
                    p_start, p_end = v.period()
                    if p_start <= d < p_end:
                        # проверка лимита бонусов
                        if max_bonus_per_person is not None and bonus_count.get(v.name, 0) >= max_bonus_per_person:
                            pass
                        else:
                            cal[ds].append(Appointment(date=ds, time=slot_time, name=v.name, is_bonus=True, is_manual=False))
                            bonus_count[v.name] = bonus_count.get(v.name, 0) + 1
                            assigned = True
                            idx += 1
                            break
                    idx += 1
                    looped += 1
                if not assigned:
                    # если никому нельзя дать бонус — прекращаем попытки для этого дня
                    break

    def export_csv(self, cal: Dict[str, List[Appointment]], out_path: str):
        rows = []
        for day, apps in sorted(cal.items()):
            for ap in sorted(apps, key=lambda x: x.time):
                rows.append([day, ap.time, ap.name, "bonus" if ap.is_bonus else ("manual" if ap.is_manual else "required")])
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(["date", "time", "client", "type"])
            writer.writerows(rows)

    def export_ics(self, cal: Dict[str, List[Appointment]], out_path: str):
        # Пишем минимальный ICS вручную
        # Каждый слот — отдельный VEVENT (DTSTART/DTEND в локальном времени без TZID — многие календари корректно поймут).
        # Для точной TZ можно добавить TZID, но это усложнит файл. Оставим простую локальную форму.
        def dt_join(d_str: str, t_str: str) -> datetime:
            d = parse_date(d_str)
            h, m = map(int, t_str.split(":"))
            return datetime(d.year, d.month, d.day, h, m, 0)

        with open(out_path, "w", encoding="utf-8") as f:
            f.write("BEGIN:VCALENDAR\r\n")
            f.write("VERSION:2.0\r\n")
            f.write("PRODID:-//Salon Scheduler//RU//EN\r\n")
            for day, apps in sorted(cal.items()):
                for ap in apps:
                    start_dt = dt_join(ap.date, ap.time)
                    end_dt = start_dt + timedelta(minutes=60)  # ровно длительность сеанса (пауза не входит)
                    uid = f"{ap.date}-{ap.time}-{ap.name}".replace(" ", "_")
                    summary = f"{ap.name} ({'БОНУС' if ap.is_bonus else ('РУЧНАЯ' if ap.is_manual else 'ОБЯЗАТ.')})"
                    f.write("BEGIN:VEVENT\r\n")
                    f.write(f"UID:{uid}\r\n")
                    f.write(f"DTSTART:{start_dt.strftime('%Y%m%dT%H%M%S')}\r\n")
                    f.write(f"DTEND:{end_dt.strftime('%Y%m%dT%H%M%S')}\r\n")
                    f.write(f"SUMMARY:{summary}\r\n")
                    f.write("END:VEVENT\r\n")
            f.write("END:VCALENDAR\r\n")

    # --------------------------
    # Публичные операции (CLI)
    # --------------------------
    def cmd_add_visitor(self, name: str, start_date_str: str, months: int = 6, min_sessions: int = 10):
        visitors = self.store.load_visitors()
        if any(v.name == name for v in visitors):
            print(f"Посетитель '{name}' уже существует.", file=sys.stderr)
            return
        # валидация даты
        _ = parse_date(start_date_str)
        visitors.append(Visitor(name=name, start_date=start_date_str, months=months, min_sessions=min_sessions))
        self.store.save_visitors(visitors)
        print(f"OK: добавлен посетитель '{name}' с началом {start_date_str}, {months} мес., минимум {min_sessions} сеансов.")

    def cmd_remove_visitor(self, name: str):
        visitors = self.store.load_visitors()
        new = [v for v in visitors if v.name != name]
        if len(new) == len(visitors):
            print(f"Посетитель '{name}' не найден.", file=sys.stderr)
            return
        self.store.save_visitors(new)
        print(f"OK: удалён посетитель '{name}'.")

    def cmd_list_visitors(self):
        visitors = self.store.load_visitors()
        if not visitors:
            print("Список пуст.")
            return
        for v in sorted(visitors, key=lambda x: (x.start_date, x.name)):
            p0, p1 = v.period()
            print(f"- {v.name}: старт {v.start_date}, период {p0}..{p1} (мин. {v.min_sessions})")

    def cmd_block_slot(self, date_str: str, time_str: str, name: Optional[str] = None, comment: Optional[str] = None):
        # Ручная запись/блокировка: ставится до автопланирования
        blocks = self.store.load_manual()
        # валидации
        _ = parse_date(date_str)
        _ = parse_time(time_str)
        blocks.append(ManualBlock(date=date_str, time=time_str, name=name, comment=comment))
        self.store.save_manual(blocks)
        who = name if name else "BLOCKED"
        print(f"OK: вручную забронирован слот {date_str} {time_str} для {who}.")

    def cmd_clear_manual(self, date_str: Optional[str] = None):
        # Полностью очистить ручные записи, либо только конкретную дату
        blocks = self.store.load_manual()
        if date_str:
            blocks = [b for b in blocks if b.date != date_str]
            print(f"OK: удалены ручные записи на дату {date_str}.")
        else:
            blocks = []
            print("OK: все ручные записи удалены.")
        self.store.save_manual(blocks)

    def cmd_generate(self, start_date_str: str, end_date_str: str, fill_bonus: bool = False, max_bonus_per_person: Optional[int] = None):
        start = parse_date(start_date_str)
        end   = parse_date(end_date_str)
        if end <= start:
            print("Ошибка: end_date должен быть позже start_date.", file=sys.stderr)
            return

        visitors = self.store.load_visitors()
        manual   = self.store.load_manual()

        cal = self.build_empty_calendar(start, end)
        # сначала проставим ручные записи
        self.place_manual(cal, manual)
        # затем обязательные 10
        self.schedule_minimums(cal, visitors, start, end)
        # и бонусы, если включено
        if fill_bonus:
            self.schedule_bonuses(cal, visitors, start, end, max_bonus_per_person=max_bonus_per_person)

        self.store.save_schedule(cal)
        print(f"OK: расписание сформировано на период {start}..{end}. Всего дней: {(end-start).days}")

    def cmd_export(self, out_csv: Optional[str], out_ics: Optional[str]):
        cal = self.store.load_schedule()
        if not cal:
            print("Сначала сгенерируйте расписание (команда generate).", file=sys.stderr)
            return
        if out_csv:
            self.export_csv(cal, out_csv)
            print(f"OK: экспорт CSV -> {out_csv}")
        if out_ics:
            self.export_ics(cal, out_ics)
            print(f"OK: экспорт ICS -> {out_ics}")

# ------------------------------
# CLI
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description="Salon Scheduler — авто-расписание для салона красоты.")
    parser.add_argument("--data-dir", default="data", help="Каталог данных (по умолчанию: ./data)")

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_add = sub.add_parser("add-visitor", help="Добавить посетителя")
    p_add.add_argument("--name", required=True)
    p_add.add_argument("--start", required=True, help="YYYY-MM-DD — дата начала абонемента (6 месяцев)")
    p_add.add_argument("--months", type=int, default=6)
    p_add.add_argument("--min", type=int, default=10, dest="min_sessions")

    p_rm = sub.add_parser("remove-visitor", help="Удалить посетителя")
    p_rm.add_argument("--name", required=True)

    p_ls = sub.add_parser("list-visitors", help="Список посетителей")

    p_blk = sub.add_parser("block-slot", help="Ручная запись/блокировка слота (ставится до автопланирования)")
    p_blk.add_argument("--date", required=True, help="YYYY-MM-DD")
    p_blk.add_argument("--time", required=True, help="HH:MM (начало слота, напр. 09:10)")
    p_blk.add_argument("--name", help="Имя клиента (если указать, запись засчитается этому клиенту)")
    p_blk.add_argument("--comment", help="Комментарий")

    p_clr = sub.add_parser("clear-manual", help="Удалить ручные записи (все или за конкретную дату)")
    p_clr.add_argument("--date", help="Если указать YYYY-MM-DD — удалит только этот день")

    p_gen = sub.add_parser("generate", help="Сгенерировать расписание на период")
    p_gen.add_argument("--start", required=True, help="YYYY-MM-DD — начало периода (включительно)")
    p_gen.add_argument("--end", required=True, help="YYYY-MM-DD — конец периода (не включительно)")
    p_gen.add_argument("--fill-bonus", action="store_true", help="Заполнить свободные слоты бонусами")
    p_gen.add_argument("--max-bonus-per-person", type=int, help="Ограничение бонусов на одного человека (опционально)")

    p_exp = sub.add_parser("export", help="Экспорт последнего сгенерированного расписания")
    p_exp.add_argument("--csv", help="Путь к CSV-файлу")
    p_exp.add_argument("--ics", help="Путь к ICS-файлу")

    args = parser.parse_args()
    store = Store(args.data_dir)
    sched = Scheduler(store)

    if args.cmd == "add-visitor":
        sched.cmd_add_visitor(args.name, args.start, months=args.months, min_sessions=args.min_sessions)
    elif args.cmd == "remove-visitor":
        sched.cmd_remove_visitor(args.name)
    elif args.cmd == "list-visitors":
        sched.cmd_list_visitors()
    elif args.cmd == "block-slot":
        sched.cmd_block_slot(args.date, args.time, name=args.name, comment=args.comment)
    elif args.cmd == "clear-manual":
        sched.cmd_clear_manual(args.date)
    elif args.cmd == "generate":
        sched.cmd_generate(args.start, args.end, fill_bonus=args.fill_bonus, max_bonus_per_person=args.max_bonus_per_person)
    elif args.cmd == "export":
        sched.cmd_export(args.csv, args.ics)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
