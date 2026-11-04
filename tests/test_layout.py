from pathlib import Path
import tempfile
import unittest

from mosim_layout import PanelLayout,panel_geometry
from crosim_agent.objectives import ObjectiveController,GUIDANCE_SCALE
from crosim_agent.features import Scene


class LayoutTests(unittest.TestCase):
    def test_dragging_one_title_bar_moves_only_that_panel_and_survives_restart(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            path = Path(folder)/"layout.json"
            layout = PanelLayout(path)
            geometry = panel_geometry(1920,1080,{})
            x,y,_,_ = geometry["minimap"]
            self.assertEqual(layout.begin(x+15,y+12,geometry),"minimap")
            layout.move(200,150,1920,1080)
            layout.end()
            loaded = PanelLayout(path)
            placed = panel_geometry(1920,1080,{},positions=loaded.positions)
            self.assertEqual(placed["reward"],geometry["reward"])
            self.assertAlmostEqual(placed["minimap"][0],185)
            self.assertAlmostEqual(placed["minimap"][1],138)

    def test_body_clicks_do_not_start_a_drag_and_offscreen_moves_are_clamped(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            layout = PanelLayout(Path(folder)/"layout.json")
            geometry = panel_geometry(1280,720,{})
            x,y,_,_ = geometry["reward"]
            self.assertIsNone(layout.begin(x+20,y+100,geometry))
            self.assertEqual(layout.begin(x+20,y+10,geometry),"reward")
            layout.move(-500,5000,1280,720)
            placed = panel_geometry(1280,720,{},positions=layout.positions)["reward"]
            self.assertEqual(placed[0],0)
            self.assertLessEqual(placed[1]+placed[3],720)

    def test_guidelines_are_weak_and_do_not_force_shooting_or_recovery(self):
        scene = Scene(hub_visible=True,hub_scale=.5,hub_x=0)
        decision = ObjectiveController().update(scene,fill=.95)
        self.assertEqual(decision.features()["goal_shoot"],GUIDANCE_SCALE)
        self.assertEqual(sum(decision.allowed),18)
        scene.hub_visible = False
        recovery = ObjectiveController().update(scene,fill=0,no_progress=10)
        self.assertEqual(recovery.goal,"recover")
        self.assertEqual(sum(recovery.allowed),18)
