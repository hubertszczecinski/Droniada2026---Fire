"""Okno edycji wyniku panelu + przycisk Wyślij (reset migawek)."""
from __future__ import annotations

import tkinter as tk
from tkinter import scrolledtext
from typing import Any, Callable, Dict, List, Optional


class PanelReviewUI:
    """Tkinter — edycja raportu; Wyślij wywołuje callback (bez wysyłki sieciowej)."""

    def __init__(
        self,
        on_send: Callable[[str, List[str]], None],
        *,
        title: str = 'Droniada — wynik panelu',
    ) -> None:
        self._on_send = on_send
        self._user_editing = False
        self._bound_panel = 'A'
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry('520x420')
        self._status = tk.StringVar(value='')
        tk.Label(self.root, textvariable=self._status, anchor='w').pack(
            fill='x', padx=8, pady=(8, 4),
        )
        tk.Label(
            self.root,
            text='Edytuj linie raportu (Wiersz/Kolumna/Kolor). Wyślij = zapis szkicu + zeruje migawki panelu.',
            wraplength=480,
            justify='left',
        ).pack(fill='x', padx=8)
        self._text = scrolledtext.ScrolledText(self.root, height=16, width=62, font=('Menlo', 11))
        self._text.pack(fill='both', expand=True, padx=8, pady=6)
        self._text.bind('<<Modified>>', self._on_text_modified)
        frm = tk.Frame(self.root)
        frm.pack(fill='x', padx=8, pady=(4, 10))
        self._btn_send = tk.Button(
            frm, text='Wyślij (reset migawek)', command=self._click_send,
            bg='#2d6a4f', fg='white', activebackground='#40916c', padx=12, pady=6,
        )
        self._btn_send.pack(side='left')
        tk.Button(frm, text='Odśwież z konkursu', command=self._mark_external_update).pack(
            side='left', padx=8,
        )
        self._external_lines: List[str] = []

    def _on_text_modified(self, _event: Any = None) -> None:
        if self._text.edit_modified():
            self._user_editing = True
            self._text.edit_modified(False)

    def _mark_external_update(self) -> None:
        self._user_editing = False
        self.set_report_lines(self._external_lines)

    def pump(self) -> None:
        try:
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            pass

    def set_status(
        self,
        *,
        panel_id: str,
        snap_n: int,
        snap_max: int,
        mission_index: int,
        mission_total: int,
        panel_full: bool,
        mission_done: bool,
    ) -> None:
        part = f'Panel {panel_id} ({mission_index + 1}/{mission_total})'
        snaps = f'Migawki: {snap_n}/{snap_max}'
        if mission_done:
            extra = ' — misja zakończona'
        elif panel_full:
            extra = ' — PEŁNY: wyślij aby przejść dalej'
        else:
            extra = ' — zbieranie migawek…'
        self._status.set(f'{part}  |  {snaps}{extra}')

    def set_report_lines(self, lines: List[str], *, force: bool = False) -> None:
        self._external_lines = list(lines)
        if self._user_editing and not force:
            return
        content = '\n'.join(lines)
        cur = self._text.get('1.0', 'end-1c')
        if cur == content:
            return
        self._text.delete('1.0', 'end')
        self._text.insert('1.0', content)
        self._text.edit_modified(False)

    def get_report_lines(self) -> List[str]:
        raw = self._text.get('1.0', 'end-1c')
        return [ln.strip() for ln in raw.splitlines() if ln.strip()]

    def _click_send(self) -> None:
        from module_panel.competition_report import validate_competition_report_lines
        from tkinter import messagebox

        lines = self.get_report_lines()
        ok, errors = validate_competition_report_lines(
            lines,
            min_cards=0,
            max_cards=4,
            expected_panel=self._bound_panel,
            allow_empty=True,
        )
        if not ok:
            messagebox.showerror(
                'Raport niepoprawny',
                'Popraw raport przed wysyłką:\n\n' + '\n'.join(errors),
            )
            return
        self._on_send(self._bound_panel, lines)
        self._user_editing = False

    def bind_panel_id(self, panel_id: str) -> None:
        self._bound_panel = str(panel_id).upper()

    def send_for_panel(self, panel_id: str) -> None:
        self._on_send(panel_id, self.get_report_lines())
