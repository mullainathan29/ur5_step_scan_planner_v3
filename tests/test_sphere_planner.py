import tempfile
from pathlib import Path

import cadquery as cq
import numpy as np

from cad_model import load_step_model
from face_analysis import SurfaceType
from planners.base import PlannerSettings, SphereStrategy
from planners.sphere import plan_spherical_face


def _load(shape):
    temp=tempfile.TemporaryDirectory(); path=Path(temp.name)/"sphere.step"
    cq.exporters.export(shape,str(path),exportType="STEP")
    model=load_step_model(path)
    i=next(r.face_id for r in model.faces if r.geometry_type is SurfaceType.SPHERE)
    return temp,model,model.topods_faces[i],model.faces[i]


def _check(plan, spacing):
    assert plan.path_segments
    for motion in plan.motion_segments:
        q=np.asarray([p.quaternion_xyzw for p in motion.scan_poses])
        if len(q)>1: assert np.all(np.sum(q[:-1]*q[1:],axis=1)>0)
        for p in motion.scan_poses:
            assert np.isclose(np.linalg.norm(p.position-p.surface_point),310)
        for a,b in zip(motion.scan_poses,motion.scan_poses[1:]):
            assert np.linalg.norm(b.surface_point-a.surface_point)<=1.5*spacing


def test_full_sphere_avoids_poles_and_is_continuous():
    temp,model,face,record=_load(cq.Solid.makeSphere(8,angleDegrees1=-90,angleDegrees2=90))
    settings=PlannerSettings(lane_spacing_mm=4,travel_spacing_mm=2)
    plan=plan_spherical_face(face,record,settings)
    _check(plan,2)
    assert all(abs(s.uv[1]) < np.pi/2 for p in plan.path_segments for s in p.samples)
    temp.cleanup()


def test_hemisphere_and_longitude_strategy():
    temp,model,face,record=_load(cq.Solid.makeSphere(8))
    settings=PlannerSettings(lane_spacing_mm=4,travel_spacing_mm=2)
    plan=plan_spherical_face(face,record,settings,strategy=SphereStrategy.LONGITUDE)
    assert plan.strategy is SphereStrategy.LONGITUDE
    _check(plan,2)
    temp.cleanup()


def test_internal_spherical_cavity_is_reversed():
    block=cq.Workplane("XY").box(20,20,20).val()
    cavity=cq.Solid.makeSphere(5,angleDegrees1=-90,angleDegrees2=90)
    temp,model,face,record=_load(block.cut(cavity))
    assert record.reversed_orientation
    plan=plan_spherical_face(face,record,PlannerSettings(lane_spacing_mm=3,travel_spacing_mm=2))
    assert plan.path_segments
    temp.cleanup()
