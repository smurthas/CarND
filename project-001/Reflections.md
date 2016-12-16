# Project 1 Reflection

Q: "How could you imagine making your algorithm better / more robust? Where will
your current algorithm be likely to fail?"

Currently, the masked area that is used to search for lane lines is hardcoded to
work well for this project. As a result, it does not adapt to curves or shifts
within the lane particularly well. Also as a result, it would likely lose track
of the lane lines during a lane change or any circumstance in which the vehicle
has a meaningful yaw angle (like sliding or spining), or in more complex
lane scenarios such as intersections. It would like also fail if a vehicle ahead
changed lanes and partially occluded the lane lines.

The strategy I employed fits regression line to the output of the Hough lines,
which works well for identifying straight lanes. I experimented with a
polynomial fit and was able to get results that, to me, demonstrated some
promise as a technique, however they were too noisy to be useful (you can them
them setting the debug flag to True on the `process_video` function calls).

My algorithm might benefit from some smoothing from frame to frame. This could
help with the occlusion issue and might also help reduce the noisiness of the
detection of dash lane lines.

