"""Detector de caídas – app Android (Kivy + LSTM en NumPy).
Pestañas: Detección (en vivo con el sensor del celular, o desde un CSV de Phyphox) y Modelos.
Sensor: aceleración lineal (sin gravedad) = lo mismo que "Acceleration without g" de Phyphox,
que es con lo que se entrenó el modelo."""
import os, json, math, time, threading, collections
import numpy as np
from kivy.app import App
from kivy.clock import Clock, mainthread
from kivy.graphics import Color, Ellipse, Line, Rectangle
from kivy.lang import Builder
from kivy.utils import platform
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.widget import Widget

import nucleo as N

KV = """
<Raiz>:
    orientation: 'vertical'
    TabbedPanel:
        do_default_tab: False
        tab_width: self.width / 2
        TabbedPanelItem:
            text: 'Detección'
            BoxLayout:
                orientation: 'vertical'
                padding: dp(8)
                spacing: dp(6)
                Label:
                    id: lbl_res
                    text: 'Listo'
                    font_size: '24sp'
                    bold: True
                    size_hint_y: None
                    height: dp(44)
                Label:
                    id: lbl_det
                    text: ''
                    size_hint_y: None
                    height: dp(36)
                    text_size: self.width, None
                    halign: 'center'
                Icono:
                    id: icono
                    size_hint_y: 0.34
                Grafica:
                    id: graf
                    size_hint_y: 0.28
                BoxLayout:
                    size_hint_y: None
                    height: dp(52)
                    spacing: dp(6)
                    ToggleButton:
                        id: btn_vivo
                        text: 'Iniciar monitoreo'
                        on_release: root.alternar_vivo()
                    Button:
                        text: 'Cargar CSV'
                        on_release: root.elegir_csv()
                    Button:
                        text: 'Limpiar'
                        on_release: root.limpiar()
                Label:
                    id: lbl_ev
                    text: 'Sin eventos'
                    size_hint_y: None
                    height: dp(70)
                    text_size: self.width, self.height
                    valign: 'top'
        TabbedPanelItem:
            text: 'Modelos'
            ScrollView:
                Label:
                    id: lbl_mod
                    size_hint_y: None
                    height: self.texture_size[1]
                    text_size: self.width, None
                    padding: dp(10), dp(10)
                    markup: True
"""


class Icono(Widget):
    modo = "idle"; fase = 0

    def tick(self, dt):
        self.fase += 1; f = self.fase; self.canvas.clear(); cx, cy = self.center_x, self.center_y
        s = min(self.width, self.height) / 2.2
        with self.canvas:
            if self.modo == "caida":
                r = s * (0.9 + 0.1 * math.sin(f / 3))
                Color(1, .6, .6) if f % 10 < 5 else Color(1, .84, .84)
                Ellipse(pos=(cx - r, cy - r), size=(2 * r, 2 * r))
                Color(.2, .2, .2)
                ang = math.radians(min(f % 40, 25) / 25 * 80)
                x0, y0 = cx, cy + s * .1; dx, dy = math.sin(ang) * s * .5, math.cos(ang) * s * .5
                Ellipse(pos=(x0 + dx - s * .1, y0 + dy + s * .1), size=(s * .2, s * .2))
                Line(points=[x0 + dx, y0 + dy, x0, y0 - s * .4], width=3)
            elif self.modo == "ok":
                Color(.85, .95, .88); Ellipse(pos=(cx - s, cy - s), size=(2 * s, 2 * s))
                Color(.2, .2, .2); Ellipse(pos=(cx - s * .1, cy + s * .35), size=(s * .2, s * .2))
                Line(points=[cx, cy + s * .35, cx, cy - s * .2], width=3)
                Line(points=[cx, cy - s * .2, cx - s * .2, cy - s * .6], width=3)
                Line(points=[cx, cy - s * .2, cx + s * .2, cy - s * .6], width=3)
            else:
                Color(.9, .9, .9); Ellipse(pos=(cx - s, cy - s), size=(2 * s, 2 * s))


class Grafica(Widget):
    """|a| en el tiempo (sin dependencias de matplotlib, que pesa mucho en Android)."""
    t = np.array([]); a = np.array([]); marcas = []

    def poner(self, t, a, marcas=()):
        self.t, self.a, self.marcas = np.asarray(t), np.asarray(a), list(marcas); self.redibujar()

    def redibujar(self, *_):
        self.canvas.clear()
        with self.canvas:
            Color(.12, .12, .12); Rectangle(pos=self.pos, size=self.size)
            if len(self.t) < 2: return
            t0, t1 = self.t[0], self.t[-1]; amax = max(float(self.a.max()), 1.0)
            fx = lambda x: self.x + (x - t0) / max(t1 - t0, 1e-9) * self.width
            fy = lambda y: self.y + 4 + y / amax * (self.height - 8)
            Color(1, 0, 0, .35)
            for a, b in self.marcas:
                Rectangle(pos=(fx(max(a, t0)), self.y), size=(max(fx(min(b, t1)) - fx(max(a, t0)), 2), self.height))
            Color(.3, .7, 1)
            k = max(1, len(self.t) // 600)
            Line(points=[c for x, y in zip(self.t[::k], self.a[::k]) for c in (fx(x), fy(y))], width=1.2)


class Raiz(BoxLayout):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.buf = collections.deque(maxlen=60 * 100)      # últimos ~60 s a hasta 100 Hz
        self.vivo = False; self.modelo = None; self.ultimo_evento = -99.0; self.eventos = []
        self.sensor = None; self.t_ini = None
        Clock.schedule_interval(self.ids.icono.tick, 1 / 12)
        self.mostrar_modelos()
        threading.Thread(target=self._cargar_modelo, daemon=True).start()

    def _cargar_modelo(self):
        self.modelo = N.ModeloLSTM()

    # ---------------- pestaña Modelos
    def mostrar_modelos(self):
        try:
            r = json.load(open(os.path.join(N.D, "resultados.json"), encoding="utf-8"))
        except Exception as e:
            self.ids.lbl_mod.text = f"No se pudo leer resultados.json\n{e}"; return
        s = [f"[b]Modelo en uso: LSTM de {r['mejor']} neuronas (*)[/b]",
             "Validación: entrenado con 2 usuarios y probado con el tercero (3 rondas, matrices sumadas).\n"]
        for f in r["tabla"]:
            s.append(f"[b]{int(f['Neuronas'])}{' *' if int(f['Neuronas']) == r['mejor'] else ''} neuronas[/b]\n"
                     f"Exactitud {f['Exactitud']:.2f} · Precisión {f['Precision']:.2f} · Recall {f['Recall']:.2f} · "
                     f"F1 {f['F1']:.2f} · Espec. {f['Especificidad']:.2f}")
        cm = r["cm"][str(r["mejor"])]
        s.append(f"\n[b]Matriz de confusión ({r['mejor']} neuronas)[/b]\n"
                 f"Real no caída → pred. no caída {cm[0][0]}, pred. caída {cm[0][1]}\n"
                 f"Real caída → pred. no caída {cm[1][0]}, pred. caída {cm[1][1]}")
        self.ids.lbl_mod.text = "\n".join(s)

    # ---------------- sensor en vivo
    def alternar_vivo(self):
        if self.ids.btn_vivo.state == "down": self.iniciar_vivo()
        else: self.detener_vivo()

    def iniciar_vivo(self):
        if platform != "android":
            self.ids.btn_vivo.state = "normal"
            self.ids.lbl_det.text = "El sensor solo existe en Android. Use «Cargar CSV» en el PC."; return
        try:
            from jnius import autoclass, PythonJavaClass, java_method, cast
            Ctx = autoclass("android.content.Context"); Act = autoclass("org.kivy.android.PythonActivity")
            Sens = autoclass("android.hardware.Sensor"); SM = autoclass("android.hardware.SensorManager")
            self._sm = cast("android.hardware.SensorManager", Act.mActivity.getSystemService(Ctx.SENSOR_SERVICE))
            lin = self._sm.getDefaultSensor(Sens.TYPE_LINEAR_ACCELERATION)
            self._gravedad = None
            if lin is None:     # respaldo: acelerómetro crudo, se le resta la gravedad con filtro paso-bajo
                lin = self._sm.getDefaultSensor(Sens.TYPE_ACCELEROMETER); self._gravedad = np.zeros(3)
            app = self

            class Oyente(PythonJavaClass):
                __javainterfaces__ = ["android/hardware/SensorEventListener"]
                __javacontext__ = "app"
                @java_method("(Landroid/hardware/Sensor;I)V")
                def onAccuracyChanged(self, s, a): pass
                @java_method("(Landroid/hardware/SensorEvent;)V")
                def onSensorChanged(self, ev):
                    v = np.array([ev.values[0], ev.values[1], ev.values[2]], float)
                    if app._gravedad is not None:
                        app._gravedad = 0.9 * app._gravedad + 0.1 * v; v = v - app._gravedad
                    app.buf.append((time.monotonic(), v[0], v[1], v[2]))

            self._oyente = Oyente()
            self._sm.registerListener(self._oyente, lin, SM.SENSOR_DELAY_GAME)   # ~50 Hz
            # mantiene la CPU activa con la pantalla apagada
            try:
                from android import mActivity  # noqa
                PM = autoclass("android.os.PowerManager"); pm = cast("android.os.PowerManager", Act.mActivity.getSystemService(Ctx.POWER_SERVICE))
                self._wl = pm.newWakeLock(PM.PARTIAL_WAKE_LOCK, "caidas:monitor"); self._wl.acquire()
            except Exception: self._wl = None
        except Exception as e:
            self.ids.btn_vivo.state = "normal"; self.ids.lbl_det.text = f"No se pudo abrir el sensor: {e}"; return
        self.buf.clear(); self.eventos = []; self.vivo = True; self.t_ini = time.monotonic()
        self.ids.btn_vivo.text = "Detener monitoreo"; self.ids.icono.modo = "ok"
        self.ids.lbl_res.text = "Monitoreando…"; self.ids.lbl_res.color = (.3, .8, .4, 1)
        Clock.schedule_interval(self.paso_vivo, N.STRIDE_S)

    def detener_vivo(self):
        Clock.unschedule(self.paso_vivo); self.vivo = False
        try: self._sm.unregisterListener(self._oyente)
        except Exception: pass
        try: self._wl.release()
        except Exception: pass
        self.ids.btn_vivo.text = "Iniciar monitoreo"; self.ids.lbl_res.text = "Monitoreo detenido"; self.ids.icono.modo = "idle"

    def paso_vivo(self, dt):
        if self.modelo is None or len(self.buf) < 50: return
        a = np.array(self.buf); t = a[:, 0] - self.t_ini; xyz = a[:, 1:4]; ahora = t[-1]
        self.ids.graf.poner(t[-1500:], np.linalg.norm(xyz[-1500:], axis=1), [(e[0], e[1]) for e in self.eventos])
        if ahora < N.WIN_S: return
        w = N.ventana(t, xyz, ahora - N.WIN_S)       # últimos 3 s -> 75 pasos a 25 Hz
        if w is None: return
        p = float(self.modelo.predecir(N.preprocesar(w[None].astype("float32")))[0])
        self.ids.lbl_det.text = f"p(caída) = {p:.2f}"
        if p >= N.THR and ahora - self.ultimo_evento > 5.0:     # un aviso por evento (enfriamiento 5 s)
            self.ultimo_evento = ahora; self.eventos.append((ahora - 3, ahora, p)); self.alarma(p, ahora)

    def alarma(self, p, t):
        self.ids.icono.modo = "caida"; self.ids.lbl_res.text = "¡CAÍDA DETECTADA!"; self.ids.lbl_res.color = (.9, .2, .2, 1)
        self.ids.lbl_ev.text = "\n".join(f"Caída {i} a los {e[1]:.0f} s (p={e[2]:.2f})" for i, e in enumerate(self.eventos[-4:], max(1, len(self.eventos) - 3)))
        try:
            from plyer import vibrator; vibrator.vibrate(1.5)
        except Exception: pass

    # ---------------- CSV de Phyphox
    def elegir_csv(self):
        try:
            from plyer import filechooser
            filechooser.open_file(on_selection=self.csv_elegido, filters=["*.csv"])
        except Exception as e:
            self.ids.lbl_det.text = f"No se pudo abrir el selector: {e}"

    @mainthread
    def csv_elegido(self, sel):
        if not sel: return
        if self.vivo: self.ids.btn_vivo.state = "normal"; self.detener_vivo()
        try:
            t, xyz = N.leer_csv(sel[0])
            while self.modelo is None: time.sleep(.05)
            res = N.clasificar(t, xyz, self.modelo); ev = N.eventos(t, xyz, res)
        except Exception as e:
            self.ids.lbl_det.text = f"Error: {e}"; return
        self.ids.graf.poner(t, np.linalg.norm(xyz, axis=1), [(a, b) for a, b, _ in ev])
        self.ids.lbl_det.text = f"{os.path.basename(sel[0])} · {len(res)} ventanas de {N.WIN_S:.0f} s"
        if ev:
            self.ids.icono.modo = "caida"; self.ids.lbl_res.text = "¡CAÍDA DETECTADA!"; self.ids.lbl_res.color = (.9, .2, .2, 1)
            self.ids.lbl_ev.text = "\n".join(f"Caída {i}: {a:.1f}–{b:.1f} s (p={p:.2f})" for i, (a, b, p) in enumerate(ev, 1))
        else:
            self.ids.icono.modo = "ok"; self.ids.lbl_res.text = "SIN CAÍDA"; self.ids.lbl_res.color = (.3, .8, .4, 1); self.ids.lbl_ev.text = "Sin eventos"

    def limpiar(self):
        if self.vivo: self.ids.btn_vivo.state = "normal"; self.detener_vivo()
        self.buf.clear(); self.eventos = []; self.ids.graf.poner([], []); self.ids.icono.modo = "idle"
        self.ids.lbl_res.text = "Listo"; self.ids.lbl_res.color = (1, 1, 1, 1); self.ids.lbl_det.text = ""; self.ids.lbl_ev.text = "Sin eventos"


class CaidasApp(App):
    title = "Detector de caídas"
    def build(self):
        Builder.load_string(KV); return Raiz()
    def on_pause(self): return True     # el monitoreo sigue con el wake lock
    def on_resume(self): pass


if __name__ == "__main__":
    CaidasApp().run()
