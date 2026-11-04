# Roadmap

The next milestone is a trustworthy observation HUD during human play, before
training a policy to play from detector features.

Implemented foundations now include zero-label hub AprilTags and a versioned
feature-policy pipeline with offline initialization. Its live-game learning and
transfer still need validation; see [training](training.md).

1. **Richer fuel regions.** Separate clear isolated balls, outline dense piles,
   and show fuel coverage without requiring an exact count of overlapping balls.
2. **Collection during watch mode.** Automatically select useful raw frames and
   failure clips from normal gameplay, without overlay contamination.
3. **Minimal manual labeling.** Test color-derived labels, isolated-ball cutouts
   and synthetic scenes, plus pretrained text/example annotation for robots and hubs.
4. **Hybrid detectors.** Support independent fuel, robot and hub backends rather
   than requiring every model to implement all four current classes.
5. **Measured promotion.** Keep real reviewed validation examples and compare
   detector candidates before replacing a working version. Generated labels can
   contain errors; self-training is not independent validation.
6. **Live HUD acceptance.** Verify useful outlines, camera-change handling, current
   frame timestamps, responsive controls and explainable rewards during human play.
7. **Playing-policy integration.** Version structured observations and recordings
   only after the HUD is reliable.

These are planned improvements, not claims about the current implementation.
