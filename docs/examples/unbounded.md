# Unbounded Experiments

In some microscopy experiments, the total number of frames to be acquired is not
known in advance. For example, a time-lapse imaging session may continue until a
specific biological event occurs, or some duration has elapsed.

`ome-writers` supports this case by allowing you to define an unbounded first
dimension in your `AcquisitionSettings`.  A dimension is considered unbounded if
its `count` is set to `None`.  Only the first dimension can be unbounded.

When using an unbounded first dimension, you can keep appending frames to the
stream until you decide to stop the acquisition.  The stream writer will handle
dynamically resizing the underlying storage as needed.

```python
--8<-- "examples/unbounded_timelapse.py"
```
