"""Independent, persistent positions for the two draggable HUD panels."""
import json
import math
from pathlib import Path
import uuid

from mosim_debug import REWARD_LABELS


LAYOUT_PATH = Path(__file__).with_name("mosim_overlay_layout.json")


def panel_geometry(width,height,state,corner="top-right",positions=None):
    labels = state.get("ledger",{}).get("labels",REWARD_LABELS)
    panel_height = 400+len(labels)*20
    scale = min(1.0,height/(panel_height+36),width/850)
    pw,ph = 440*scale,panel_height*scale
    ox = width-pw-18*scale if corner.endswith("right") else 18*scale
    oy = height-ph-18*scale if corner.startswith("bottom") else 18*scale
    mx = ox-352*scale if corner.endswith("right") else ox+pw+12*scale
    my = oy if corner.startswith("top") else oy+ph-418*scale
    result = {"reward":[ox,oy,pw,ph],"minimap":[mx,my,340*scale,418*scale],"scale":scale}
    for name in ("reward","minimap"):
        box = result[name]
        value = (positions or {}).get(name)
        if value and len(value) == 2 and all(isinstance(v,(int,float)) and math.isfinite(v) for v in value):
            box[:2] = [value[0]*width,value[1]*height]
        box[0] = min(max(0,box[0]),max(0,width-box[2]))
        box[1] = min(max(0,box[1]),max(0,height-box[3]))
    return result


class PanelLayout:
    def __init__(self,path=LAYOUT_PATH):
        self.path = Path(path)
        self.positions = {}
        self.drag = None
        try:
            data = json.loads(self.path.read_text())
            for name in ("reward","minimap"):
                value = data.get(name)
                if isinstance(value,list) and len(value) == 2 and all(isinstance(v,(int,float)) and math.isfinite(v) for v in value):
                    self.positions[name] = value
        except (OSError,ValueError,TypeError,AttributeError):
            pass

    @staticmethod
    def header_at(x,y,geometry,show_map=True):
        for name in ("minimap","reward") if show_map else ("reward",):
            left,top,width,_ = geometry[name]
            if left <= x <= left+width and top <= y <= top+36*geometry["scale"]:
                return name
        return None

    def begin(self,x,y,geometry,show_map=True):
        name = self.header_at(x,y,geometry,show_map)
        if name:
            left,top,width,height = geometry[name]
            self.drag = (name,x-left,y-top,width,height)
        return name

    def move(self,x,y,width,height):
        if self.drag:
            name,dx,dy,pw,ph = self.drag
            left = min(max(0,x-dx),max(0,width-pw))
            top = min(max(0,y-dy),max(0,height-ph))
            self.positions[name] = [left/width,top/height]

    def end(self):
        self.drag = None
        temporary = self.path.with_name(self.path.name+"."+uuid.uuid4().hex+".tmp")
        try:
            temporary.write_text(json.dumps(self.positions,indent=2))
            temporary.replace(self.path)
        except OSError:
            pass
        finally:
            temporary.unlink(missing_ok=True)
