"""Keyframe-only tracking ablations. Boxes are xyxy in detector-input pixels.

SORT's seven-state Kalman model is used without association: there are no new
detections within a clip. Each clip resets tracks (and therefore velocity).
"""
import cv2
import numpy as np


class SortState:
    """SORT's constant-velocity box model.

    State: [center_x, center_y, area, width/height, vx, vy, area_velocity].
    P is state uncertainty; F advances motion by one frame; Q adds process noise.
    One keyframe cannot establish velocity, so all velocities start at zero.
    Without later measurements, this prediction-only SORT holds boxes still.
    """

    def __init__(self, box):
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        self.x = np.array([(x1+x2)/2, (y1+y2)/2, w*h, w/h, 0, 0, 0.])
        self.P = np.eye(7) * 10
        # Initial velocity is unknown, so give it much greater uncertainty.
        self.P[4:, 4:] *= 1000
        self.F = np.eye(7)
        self.F[0, 4] = self.F[1, 5] = self.F[2, 6] = 1
        self.Q = np.eye(7)
        self.Q[4:, 4:] *= .01
        self.Q[6, 6] *= .01

    def predict(self):
        """Advance center/area by their velocities and increase uncertainty."""
        if self.x[2] + self.x[6] <= 0:
            self.x[6] = 0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.box()

    def correct_center(self, center, variance):
        """Fuse a flow-based center with the prediction using Kalman gain K.

        H observes only the center; R describes measurement uncertainty. Larger
        R gives flow less influence relative to the predicted uncertainty P.
        Position/velocity covariance lets center observations teach velocity.
        Area and aspect receive no measurements, so box size stays unchanged.
        """
        H = np.eye(7)[:2]
        R = np.eye(2) * variance
        K = np.linalg.solve(H @ self.P @ H.T + R, H @ self.P).T
        self.x += K @ (center - H @ self.x)
        A = np.eye(7) - K @ H
        # Joseph covariance update maintains numerical stability.
        self.P = A @ self.P @ A.T + K @ R @ K.T
        return self.box()

    def box(self):
        w = np.sqrt(max(self.x[2] * self.x[3], 1e-8))
        h = self.x[2] / w
        return np.array([self.x[0]-w/2, self.x[1]-h/2,
                         self.x[0]+w/2, self.x[1]+h/2])


def flow_displacement(previous, current, box, max_points=40, min_points=4,
                      fb_threshold=1.5):
    """Sparse LK with forward/backward and robust motion-consensus filtering.

    Points are redetected in the previous box each step. Returns (dxdy, quality)
    or None; failure never creates a new track or a detector request.
    Displacements and thresholds are in resized detector-input pixels.
    """
    height, width = previous.shape
    x1, y1, x2, y2 = box
    # Exclude a 10% margin on each side to reduce background contamination.
    inset_x, inset_y = .1 * (x2-x1), .1 * (y2-y1)
    x1, x2 = np.clip([np.ceil(x1+inset_x), np.floor(x2-inset_x)], 0, width).astype(int)
    y1, y2 = np.clip([np.ceil(y1+inset_y), np.floor(y2-inset_y)], 0, height).astype(int)
    if x2-x1 < 3 or y2-y1 < 3:
        return None
    mask = np.zeros_like(previous)
    mask[y1:y2, x1:x2] = 255
    # Choose up to 40 textured corners; fewer than four cannot support our estimate.
    points = cv2.goodFeaturesToTrack(previous, maxCorners=max_points,
                                   qualityLevel=.01, minDistance=3, mask=mask)
    if points is None or len(points) < min_points:
        return None
    options = dict(winSize=(21, 21), maxLevel=3,
                   criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
    moved, status, _ = cv2.calcOpticalFlowPyrLK(previous, current, points, None, **options)
    if moved is None:
        return None
    # A reliable correspondence should return near its starting point when
    # tracked backward. This is a failure check, not explicit occlusion detection.
    back, back_status, _ = cv2.calcOpticalFlowPyrLK(current, previous, moved, None, **options)
    if back is None:
        return None
    p, q, b = points[:, 0], moved[:, 0], back[:, 0]
    valid = ((status.ravel() == 1) & (back_status.ravel() == 1)
             & np.isfinite(q).all(1) & np.isfinite(b).all(1)
             & (np.linalg.norm(p-b, axis=1) <= fb_threshold)
             & (q[:, 0] >= 0) & (q[:, 0] < width)
             & (q[:, 1] >= 0) & (q[:, 1] < height))
    delta = q[valid] - p[valid]
    if len(delta) < min_points:
        return None
    median = np.median(delta, axis=0)
    residual = np.linalg.norm(delta-median, axis=1)
    # Keep motions near the group median; require >=4 points and >=25% survival.
    inliers = residual <= max(1.5, 3 * float(np.median(residual)))
    if inliers.sum() < min_points or inliers.sum()/len(points) < .25:
        return None
    quality = float(inliers.sum()/len(points))
    # Median translation tolerates a minority of points following other surfaces.
    return np.median(delta[inliers], axis=0), quality


def track_clip(images, keyframe_detections, mode):
    """Return per-frame Nx6 [x,y,w,h,score,class] arrays in input pixels.

    Images: uint8 RGB TCHW. Scores/classes are carried unchanged from keyframe.
    Flow-only holds the previous box on failure; combined mode coasts with KF.
    Tracks live to the clip boundary, with no births or detector refreshes.

    sort: predict only (zero initial velocity means stationary boxes).
    flow: translate the previous box directly using observed median motion.
    sort_flow: predict, then correct the center using a noisy flow measurement.

    Box sizes stay fixed. No Hungarian association, new-object discovery,
    explicit occlusion handling or re-identification is performed. Each call
    resets all states; objects are handled independently within the clip.
    """
    if mode not in ('sort', 'flow', 'sort_flow'):
        raise ValueError(mode)
    detections = np.asarray(keyframe_detections, dtype=float).reshape(-1, 6).copy()
    boxes = detections[:, :4].copy()
    boxes[:, 2:] += boxes[:, :2]
    states = [SortState(box) for box in boxes] if mode != 'flow' else []
    gray = ([cv2.cvtColor(np.moveaxis(frame, 0, -1), cv2.COLOR_RGB2GRAY)
             for frame in images] if mode != 'sort' else None)
    outputs = [detections.copy()]
    for t in range(1, len(images)):
        for i, previous_box in enumerate(boxes.copy()):
            predicted = states[i].predict() if states else previous_box
            observation = (flow_displacement(gray[t-1], gray[t], previous_box)
                           if gray is not None else None)
            if observation is not None:
                displacement, quality = observation
                # Apply flow to the PREVIOUS box, not the Kalman prediction:
                # adding it to the predicted box would count motion twice.
                measured = previous_box + np.tile(displacement, 2)
                if mode == 'flow':
                    predicted = measured
                else:
                    center = (measured[:2] + measured[2:]) / 2
                    # q=1 -> variance 4; q=.5 -> 16. Less reliable flow gets
                    # less weight. Corrections also establish velocity for coasting.
                    predicted = states[i].correct_center(center, variance=4/quality**2)
            boxes[i] = predicted
        frame = detections.copy()
        frame[:, :2] = boxes[:, :2]
        frame[:, 2:4] = boxes[:, 2:] - boxes[:, :2]
        outputs.append(frame)
    return outputs
