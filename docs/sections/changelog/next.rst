Changelog: next release
=======================

Changes after v2.14.0. They are released with the next version.

**raisim**

* Segmentation cameras (``raisim::SegmentationCamera``, ``<sensor type="segmentation">``) report
  the world index, the body index and the semantic label seen in each pixel, in three ``int32``
  images with ``-1`` where nothing is hit; ``SEMANTIC``, ``INSTANCE`` and ``PANOPTIC`` types select
  them. ``MeasurementSource::RAISIM`` ray-casts the collision geometry. See :ref:`segmentation-camera`.
* Segmentation labels: ``Object::setSegmentationLabel()``, ``ArticulatedSystem::setBodySegmentationLabel()``
  and ``clearBodySegmentationLabels()``, and class names with ``World::addSegmentationClass()``,
  ``getSegmentationClassName()`` and ``setSegmentationClassName()``. World XML files, checkpoints
  and the ``.rscene`` ``semanticClass`` key carry them.
* ``SegmentationCamera::colorize()`` colors id images for display.
* raisimPy binds the segmentation camera (``(H, W)`` ``int32`` NumPy images), the labels and the
  class names.

**rayrai**

* ``RayraiWindow::renderSegmentation()`` renders segmentation cameras on the GPU, and
  ``Camera::getRawImage()`` / ``readSegmentation()`` read them back; visuals write
  ``Visuals::getSegmentationIds()``.
* **Behavior change**: a fisheye camera's ``renderDepthPlaneDistance()`` depth is remapped through
  the lens like its color image, so ``DepthCamera::depthToPointCloud()`` reconstructs its points.
  It used to stay rectilinear.

**TCP viewer**

* ``kProtocolFeatureSegmentation``: the viewer renders ``MANUAL`` segmentation cameras for a
  ``RaisimServer`` and previews them; servers send them only to viewers that negotiate the feature.

**RaiSim Engine 2**

* Segmentation camera sensors, and ``semanticClass`` labels the bodies the RaiSim bridge creates.
