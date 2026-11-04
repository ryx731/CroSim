"""Conservative estimated contact, not collision telemetry.

Nominal REBUILT footprints from FIRST manual sections 5.4-5.6. A 0.45m
robot reach expands each footprint. Every pose-bound corner must agree.
"""
from mosim_field import FIELD


def structure_contact(field_map):
    pose = field_map.get("pose") or {}
    if not field_map.get("pose_current") or pose.get("uncertainty_m",99) > 1.25:
        return None,None
    bounds = pose.get("bounds")
    if not bounds or len(bounds) != 4:
        return None,None
    def inside(rect):
        a,b,c,d = rect
        return bounds[0] >= a-.45 and bounds[1] >= b-.45 and bounds[2] <= c+.45 and bounds[3] <= d+.45
    blue_x,red_x,width = 4.6256,11.9154,FIELD['width']
    def bump(x):
        return any(inside((x-.564,y,x+.564,y+1.854)) for y in (1.668,width-1.668-1.854))
    red = bump(red_x) or inside((red_x-.6,width/2-.6,red_x+.6,width/2+.6))
    red |= any(inside((red_x-.597,y,red_x+.597,y+1.668)) for y in (0,width-1.668))
    return bool(bump(blue_x)),bool(red)
