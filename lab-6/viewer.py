"""AnyWidget bridge. Replay is entirely browser-side; execute runs Python."""
from pathlib import Path
import anywidget
import traitlets

from experiment import (HE_STARTER, FL_STARTER, INPUTS, sources, run_he, run_fl,
                        submission_card, fl_settings)


class ProtocolLab(anywidget.AnyWidget):
    _esm = Path(__file__).with_name("viewer.js").read_text()
    _css = Path(__file__).with_name("viewer.css").read_text()
    config = traitlets.Dict().tag(sync=True)
    request = traitlets.Dict().tag(sync=True)
    response = traitlets.Dict().tag(sync=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config = dict(sources=sources(), he_source=HE_STARTER,
                           fl_source=FL_STARTER, inputs=INPUTS)
        self._results = {}
        self._signature = None
        self.observe(self._execute, names="request")

    def _execute(self, change):
        req = change["new"]
        if not req:
            return
        revisions = req.get("revisions", {})
        signature = {"he": (req.get("he_source"), str(req.get("inputs")), revisions.get("he", 0)),
                     "fl": (req.get("fl_source"), req.get("num_rounds", 1), req.get("learning_rate", .1), revisions.get("fl", 0))}
        if req["action"] == "reset":
            self._results = {}
        elif self._signature:
            for part in ("he", "fl"):
                if signature[part] != self._signature[part]:
                    self._results.pop(part, None)
                    self._results.pop("linked", None)
        self._signature = signature
        action = req["action"]
        try:
            if action == "he":
                self._results["he"] = run_he(req["he_source"], req["inputs"])
            elif action in ("one", "twenty"):
                previous = self._results.get("fl")
                if previous and previous.get("error"):
                    previous = None
                current = previous["history"][-1]["round"] if previous else 0
                rounds, _ = fl_settings(req.get("num_rounds", 1), req.get("learning_rate", .1))
                target = min(rounds, current + 1) if action == "one" else rounds
                self._results["fl"] = run_fl(req["fl_source"], target, previous, rounds, req.get("learning_rate", .1))
            fl = self._results.get("fl")
            if action in ("he", "one", "twenty") and "he" in self._results and fl and not fl["error"] and fl["history"][-1]["round"] == fl.get("num_rounds", 1):
                self._results["linked"] = run_he(req["he_source"], req["inputs"],
                    weights=fl["theta"][:2], bias=fl["theta"][2])
            card = None
            if action == "card":
                if not all(k in self._results for k in ("he", "fl", "linked")):
                    raise ValueError("Finish all configured Part A rounds and execute Part B before exporting")
                card = submission_card(self._results["he"], self._results["fl"],
                                       self._results["linked"])
            self.response = dict(id=req["id"], action=action, results=self._results.copy(), card=card)
        except Exception as exc:
            self.response = dict(id=req["id"], action=action, results=self._results.copy(),
                                 error=f"{type(exc).__name__}: {exc}")
