"""Scheduling helpers.

No daemon here on purpose. A long-lived Python scheduler is another process to
supervise, restart, and monitor -- and it fails silently. cron and systemd are
already supervised by the operating system, already log, and already survive a
reboot.

The one thing worth doing in Python is deciding whether *today* is a session
worth running on, since cron cannot know the NSE holiday calendar.
"""

from __future__ import annotations

from datetime import date, time

from ..ingestion.calendar import TradingCalendar

# NSE closes at 15:30 IST. Running before that would compute features from a
# partial bar, which is a subtler error than it looks: the close is wrong and
# nothing downstream can tell.
DEFAULT_RUN_TIME = time(18, 0)


def should_run_today(
    when: date | None = None, calendar: TradingCalendar | None = None
) -> tuple[bool, str]:
    """Was ``when`` a trading session?"""
    when = when or date.today()
    calendar = calendar or TradingCalendar()

    if not calendar.is_session(when):
        return False, f"{when} is not a trading session; skipping."
    if calendar.is_approximate:
        return True, (
            f"{when} looks like a session, but the calendar is the weekday "
            "approximation. Install pandas-market-calendars before relying on "
            "this for holidays."
        )
    return True, f"{when} is a trading session."


def crontab_line(project_root: str, run_time: time = DEFAULT_RUN_TIME) -> str:
    """A cron entry for the daily job, Monday to Friday.

    Holiday handling stays in ``should_run_today``: cron fires on every weekday
    and the pipeline exits early on a holiday, which is simpler than trying to
    express the NSE calendar in a crontab.
    """
    return (
        f"{run_time.minute} {run_time.hour} * * 1-5 "
        f"cd {project_root} && "
        f"/usr/bin/env python main.py daily >> logs/cron.log 2>&1"
    )


def systemd_timer(project_root: str, run_time: time = DEFAULT_RUN_TIME) -> str:
    """A systemd timer unit, for hosts that prefer it to cron."""
    return f"""[Unit]
Description=TradeMind AI daily pipeline

[Timer]
OnCalendar=Mon..Fri {run_time.hour:02d}:{run_time.minute:02d}
Persistent=true

[Install]
WantedBy=timers.target

# Paired service unit:
# [Service]
# Type=oneshot
# WorkingDirectory={project_root}
# ExecStart=/usr/bin/env python main.py daily
"""
