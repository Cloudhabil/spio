"""
Brahim Onion Agent - Kivy Mobile App

Android APK entry point using Kivy framework.
Build with: buildozer android debug

Author: Elias Oulad Brahim
DOI: 10.5281/zenodo.18356196
"""

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.scrollview import ScrollView
from kivy.uix.spinner import Spinner
from kivy.uix.popup import Popup
from kivy.clock import Clock
from kivy.properties import StringProperty, ListProperty
from kivy.core.window import Window

import json
import threading

# Brahim SDK imports
try:
    from ..agents_sdk import (
        BRAHIM_SEQUENCE,
        SUM_CONSTANT,
        CENTER,
        PHI,
        fine_structure_constant,
        weinberg_angle,
        muon_electron_ratio,
        proton_electron_ratio,
        cosmic_fractions,
        yang_mills_mass_gap,
        mirror_operator,
        get_sequence,
        verify_mirror_symmetry,
    )
    from ..openai_agent import BrahimOnionAgent, BrahimAgentBuilder
    SDK_AVAILABLE = True
except ImportError:
    # Fallback for standalone execution
    try:
        from brahims_laws.agents_sdk import (
            BRAHIM_SEQUENCE, SUM_CONSTANT, CENTER, PHI,
            fine_structure_constant, weinberg_angle, muon_electron_ratio,
            proton_electron_ratio, cosmic_fractions, yang_mills_mass_gap,
            mirror_operator, get_sequence, verify_mirror_symmetry,
        )
        from brahims_laws.openai_agent import BrahimOnionAgent, BrahimAgentBuilder
        SDK_AVAILABLE = True
    except ImportError:
        SDK_AVAILABLE = False
        BRAHIM_SEQUENCE = [27, 42, 60, 75, 97, 117, 139, 154, 172, 187]  # Corrected 2026-01-26
        SUM_CONSTANT = 214
        CENTER = 107
        PHI = 1.618033988749895


# =============================================================================
# MAIN APP WIDGET
# =============================================================================

class BrahimAgentWidget(BoxLayout):
    """Main widget for Brahim Onion Agent."""

    result_text = StringProperty("")
    history = ListProperty([])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = "vertical"
        self.padding = 10
        self.spacing = 10

        # Initialize agent
        if SDK_AVAILABLE:
            self.agent = BrahimAgentBuilder().with_name("mobile-agent").build()
        else:
            self.agent = None

        self._build_ui()

    def _build_ui(self):
        """Build the UI components."""

        # Header
        header = BoxLayout(size_hint_y=None, height=60)
        header.add_widget(Label(
            text="[b]BRAHIM ONION AGENT[/b]",
            markup=True,
            font_size=24
        ))
        self.add_widget(header)

        # Calculation selector
        calc_layout = BoxLayout(size_hint_y=None, height=50)
        calc_layout.add_widget(Label(text="Calculate:", size_hint_x=0.3))

        self.calc_spinner = Spinner(
            text="Fine Structure",
            values=[
                "Fine Structure",
                "Weinberg Angle",
                "Muon/Electron",
                "Proton/Electron",
                "Cosmology",
                "Yang-Mills",
                "Sequence",
                "Verify Axioms"
            ],
            size_hint_x=0.7
        )
        calc_layout.add_widget(self.calc_spinner)
        self.add_widget(calc_layout)

        # Calculate button
        calc_btn = Button(
            text="Calculate",
            size_hint_y=None,
            height=50,
            background_color=(0.2, 0.6, 1, 1)
        )
        calc_btn.bind(on_press=self.on_calculate)
        self.add_widget(calc_btn)

        # Chat input
        chat_layout = BoxLayout(size_hint_y=None, height=50)
        self.chat_input = TextInput(
            hint_text="Ask anything about physics...",
            multiline=False,
            size_hint_x=0.8
        )
        self.chat_input.bind(on_text_validate=self.on_chat_submit)
        chat_layout.add_widget(self.chat_input)

        send_btn = Button(text="Send", size_hint_x=0.2)
        send_btn.bind(on_press=self.on_chat_submit)
        chat_layout.add_widget(send_btn)
        self.add_widget(chat_layout)

        # Mirror operator
        mirror_layout = BoxLayout(size_hint_y=None, height=50)
        mirror_layout.add_widget(Label(text="Mirror M(x):", size_hint_x=0.3))
        self.mirror_input = TextInput(
            text="107",
            multiline=False,
            input_filter="int",
            size_hint_x=0.4
        )
        mirror_layout.add_widget(self.mirror_input)
        mirror_btn = Button(text="Apply", size_hint_x=0.3)
        mirror_btn.bind(on_press=self.on_mirror)
        mirror_layout.add_widget(mirror_btn)
        self.add_widget(mirror_layout)

        # Result area
        self.add_widget(Label(text="Result:", size_hint_y=None, height=30))

        scroll = ScrollView()
        self.result_label = Label(
            text="Select a calculation or ask a question",
            text_size=(Window.width - 20, None),
            size_hint_y=None,
            halign="left",
            valign="top"
        )
        self.result_label.bind(texture_size=self.result_label.setter("size"))
        scroll.add_widget(self.result_label)
        self.add_widget(scroll)

        # Constants display
        const_layout = BoxLayout(size_hint_y=None, height=80)
        const_layout.add_widget(Label(
            text=f"[b]Constants[/b]\nS={SUM_CONSTANT} | C={CENTER} | φ={PHI:.4f}" if SDK_AVAILABLE else "SDK not loaded",
            markup=True,
            halign="center"
        ))
        self.add_widget(const_layout)

        # Footer
        footer = Label(
            text="DOI: 10.5281/zenodo.18356196 | v1.3.0",
            size_hint_y=None,
            height=30,
            font_size=12
        )
        self.add_widget(footer)

    def on_calculate(self, instance):
        """Handle calculation button press."""
        if not SDK_AVAILABLE:
            self._show_result("SDK not available")
            return

        selection = self.calc_spinner.text

        # Run in background to avoid UI freeze
        threading.Thread(target=self._do_calculation, args=(selection,)).start()

    def _do_calculation(self, selection):
        """Perform calculation in background."""
        try:
            if selection == "Fine Structure":
                result = fine_structure_constant().to_dict()
            elif selection == "Weinberg Angle":
                result = weinberg_angle().to_dict()
            elif selection == "Muon/Electron":
                result = muon_electron_ratio().to_dict()
            elif selection == "Proton/Electron":
                result = proton_electron_ratio().to_dict()
            elif selection == "Cosmology":
                result = cosmic_fractions().to_dict()
            elif selection == "Yang-Mills":
                result = yang_mills_mass_gap().to_dict()
            elif selection == "Sequence":
                result = get_sequence()
            elif selection == "Verify Axioms":
                result = verify_mirror_symmetry()
            else:
                result = {"error": "Unknown selection"}

            # Update UI on main thread
            Clock.schedule_once(lambda dt: self._show_result(result))

        except Exception as e:
            Clock.schedule_once(lambda dt: self._show_result({"error": str(e)}))

    def on_chat_submit(self, instance):
        """Handle chat submission."""
        if not self.agent:
            self._show_result("Agent not initialized")
            return

        text = self.chat_input.text.strip()
        if not text:
            return

        self.chat_input.text = ""

        # Run in background
        threading.Thread(target=self._do_chat, args=(text,)).start()

    def _do_chat(self, text):
        """Process chat in background."""
        try:
            response = self.agent.process(text)
            result = {
                "intent": response.intent.value,
                "result": response.result,
                "layers": [l.value for l in response.layers_traversed]
            }
            Clock.schedule_once(lambda dt: self._show_result(result))
        except Exception as e:
            Clock.schedule_once(lambda dt: self._show_result({"error": str(e)}))

    def on_mirror(self, instance):
        """Handle mirror operator."""
        if not SDK_AVAILABLE:
            self._show_result("SDK not available")
            return

        try:
            value = int(self.mirror_input.text)
            result = mirror_operator(value)
            self._show_result(result)
        except ValueError:
            self._show_result({"error": "Invalid number"})

    def _show_result(self, result):
        """Display result in the UI."""
        if isinstance(result, dict):
            text = json.dumps(result, indent=2, default=str)
        else:
            text = str(result)

        self.result_label.text = text
        self.history.append(result)


# =============================================================================
# KIVY APP
# =============================================================================

class BrahimOnionAgentApp(App):
    """Main Kivy application."""

    def build(self):
        self.title = "Brahim Onion Agent"
        Window.clearcolor = (0.1, 0.1, 0.15, 1)
        return BrahimAgentWidget()

    def on_start(self):
        """Called when app starts."""
        print("Brahim Onion Agent started")
        print(f"Sequence: {BRAHIM_SEQUENCE}" if SDK_AVAILABLE else "SDK not loaded")

    def on_stop(self):
        """Called when app stops."""
        print("Brahim Onion Agent stopped")


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    """Run the mobile app."""
    BrahimOnionAgentApp().run()


if __name__ == "__main__":
    main()
