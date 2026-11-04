"""Shared Tk/Pillow field-map painter; all positions are estimates in meters."""
import math

from mosim_field import FIELD


def paint_minimap(p,x,y,scale,data,now):
    bg,border,fg,muted,cyan,red = "#101924","#304154","#edf4fa","#91a5b8","#6cd5f7","#ff8291"
    def rect(a,b,c,d,fill=None,outline=None):
        p.rect((x+a*scale,y+b*scale,x+c*scale,y+d*scale),fill,outline)
    def text(a,b,value,size=10,color=fg,bold=False):
        p.text(x+a*scale,y+b*scale,value,max(8,int(size*scale)),color,"nw",bold,False)
    def line(points,color):
        p.line([(x+a*scale,y+b*scale) for a,b in points],color)
    field = data.get("field") or FIELD
    length,width = field["length"],field["width"]
    fx,fy,fw,fh = 14,43,312,152
    def point(px,py):
        return fx+min(length,max(0,px))/length*fw,fy+(1-min(width,max(0,py))/width)*fh
    rect(0,0,340,305,bg,border)
    rect(0,0,3,305,cyan)
    text(14,11,"FIELD MAP / ESTIMATED POSE",12,bold=True)
    text(310,12,"drag",8,muted)
    text(14,28,"Yellow: fuel / opacity = geometric confidence",9,muted)
    blue_end,red_start = field["blue_end"],field["red_start"]
    bx,rx = point(blue_end,0)[0],point(red_start,0)[0]
    rect(fx,fy,bx,fy+fh,"#183b61")
    rect(bx,fy,rx,fy+fh,"#1c2937")
    rect(rx,fy,fx+fw,fy+fh,"#552b37")
    rect(fx,fy,fx+fw,fy+fh,None,border)
    for blob in data.get('fuel_clusters',[]):
        age = now-blob.get('captured_at',0)
        if not 0 <= age <= 2: continue
        px,py = point(blob['x'],blob['y'])
        radius = blob.get('radius',.5)
        rx,ry = max(3,radius/length*fw),max(3,radius/width*fh)
        box = (x+max(fx,px-rx)*scale,y+max(fy,py-ry)*scale,
               x+min(fx+fw,px+rx)*scale,y+min(fy+fh,py+ry)*scale)
        alpha = (.05+.85*max(0,min(1,blob.get('confidence',0)))**2)*(1-age/2)
        if hasattr(p,'blob'):
            p.blob(box,alpha)
        else:
            p.rect(box,'#857345',None)
    for at,color in ((blue_end,cyan),(length/2,border),(red_start,red)):
        px = point(at,0)[0]
        line([(px,fy),(px,fy+fh)],color)
    text(fx+8,fy+6,"BLUE",9,cyan,True)
    text(fx+fw/2-23,fy+6,"NEUTRAL",8,muted)
    text(fx+fw-48,fy+6,"RED",9,red,True)
    for hx,color in ((4.6256,cyan),(11.9154,red)):
        px,py = point(hx,width/2)
        rect(px-5,py-5,px+5,py+5,None,color)
        text(px-9,py+8,"HUB",8,color)
    trail = [point(*v) for v in data.get("trail",[])]
    if len(trail) >= 2:
        line(trail,muted)
    for robot in data.get("robots",[]):
        px,py = point(robot["x"],robot["y"])
        color = cyan if robot.get("alliance") == "blue" else red
        rect(px-3,py-3,px+3,py+3,color)
    pose = data.get("pose")
    if not pose:
        from mosim_field import FieldPoseTracker
        pose = FieldPoseTracker().pose
    age = max(0.0,now-pose["captured_at"]) if pose else float("inf")
    current = data.get("pose_current",False) and age <= .4
    if pose:
        color = cyan if current else muted
        lowx,lowy,highx,highy = pose["bounds"]
        a,b = point(lowx,highy)
        c,d = point(highx,lowy)
        rect(a,b,c,d,None,muted)
        px,py = point(pose["x"],pose["y"])
        angle = pose["heading"]
        dx,dy = math.cos(angle),-math.sin(angle)
        arrow = [(px+10*dx,py+10*dy),(px-6*dx-5*dy,py-6*dy+5*dx),
                 (px-6*dx+5*dy,py-6*dy-5*dx)]
        line([*arrow,arrow[0]],color)
        text(14,207,f"x {pose['x']:.1f}m   y {pose['y']:.1f}m   heading ~{math.degrees(angle)%360:.0f} deg",10,color)
        zone = data.get("zone","estimated") if current else "last pose held"
        text(14,228,f"Uncertainty +/-{pose['uncertainty_m']:.1f}m  |  {zone}",10,red if zone == "red" else muted)
    text(14,249,"Vision correction + retained pose; no gyro feed",9,muted)
    cost = data.get("zone_cost")
    if cost:
        text(14,269,f"Red: -{cost['entry']:.2f} entry  /  -{cost['per_second']:.2f} per second",10,red)
        if data.get("opponent_zone") is None or not current:
            text(14,288,"Zone cost paused: pose or boundary uncertain",8,muted)
    else:
        text(14,269,"Triangle: your robot   Squares: detected robots",9,muted)
        text(14,288,"Outline: uncertainty   Trail: observed estimates",8,muted)
    rect(0,315,340,418,bg,border)
    text(14,325,"FIELD STATE",10,muted,True)
    location = data.get("location") or {}
    region = location.get("region") or ("blue" if pose["x"] < blue_end else "red" if pose["x"] > red_start else "neutral")
    label = location.get("label") or {"blue":"BLUE ALLIANCE","neutral":"NEUTRAL FIELD","red":"RED ALLIANCE"}[region]
    text(14,342,label,16,red if region == "red" else cyan if region == "blue" else fg,True)
    source = location.get("source","vision estimate" if current else "last region held")
    text(14,364,source,9,muted)
    shift = data.get("shift_on")
    text(14,382,"SHIFT ON: return to blue" if shift is True else "SHIFT OFF: leave blue" if shift is False else "Shift arrow: waiting for a clear view",10,cyan if shift is True else muted)
    cost = data.get("shift_zone_per_second")
    if cost is not None:
        paused = shift is None or data.get("blue_zone") is None or not current
        text(14,400,f"Wrong-zone -{cost:.2f}/s / "+("paused: uncertain sensors" if paused else "2s travel grace"),8,muted)
