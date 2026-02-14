#############################
Math Classes
#############################

RaiSim provides small, header-only matrix/vector types that are compatible with
Eigen via lightweight ``Eigen::Map`` views.

Core types
=============================
* ``raisim::Mat<n, m>``: fixed-size, column-major storage (``double``).
* ``raisim::Vec<n>``: alias for ``raisim::Mat<n, 1>``.
* ``raisim::MatDyn`` / ``raisim::VecDyn``: dynamic-size, RaiSim-owned storage.
* ``raisim::SparseJacobian`` / ``raisim::SparseJacobian1D``: dynamic storage
  used by some APIs for sparse Jacobians.

Eigen interoperability (``.e()``)
================================================
All RaiSim math types provide an ``.e()`` accessor returning an ``Eigen::Map``
that references the underlying RaiSim storage. This is the preferred way to
call APIs that take Eigen types (e.g. ``Eigen::Ref``).

.. code-block:: cpp

  raisim::Vec<3> v;
  v << 1.0, 2.0, 3.0;

  // Non-owning Eigen view
  auto v_e = v.e();  // Eigen::Map<...>
  Eigen::Vector3d ev = v_e;  // copies into a real Eigen vector

  raisim::VecDyn q(12);
  q.setZero();
  auto q_e = q.e();  // Eigen::Map

Alignment and ownership
=============================
* ``raisim::Mat``/``raisim::Vec`` are 32-byte aligned fixed-size POD-like types.
  If you store them in standard containers, use a modern compiler/STL that
  supports over-aligned allocation (C++17 or newer recommended).
* ``raisim::MatDyn``/``raisim::VecDyn`` allocate 32-byte aligned storage
  internally. Do not free their memory manually.
* ``.e()`` returns a non-owning map. Its lifetime must not exceed the underlying
  RaiSim object.
* Any call to ``resize()`` (dynamic types) invalidates raw pointers and Eigen maps.

Initialization and indexing
=============================
``raisim::Mat`` and ``raisim::Vec`` store data in **column-major** order.

.. code-block:: cpp

  raisim::Mat<3,3> A;
  A.setZero();
  A(0,0) = 1.0;
  A(1,1) = 1.0;
  A(2,2) = 1.0;

  // Eigen-style comma initializer (fills linear storage; column-major)
  raisim::Vec<3> x;
  x << 1.0, 2.0, 3.0;

For dynamic types, prefer filling through ``.e()``:

.. code-block:: cpp

  raisim::VecDyn y(6);
  y.e().setOnes();

Const-ref guidance
=============================
To avoid copies, pass math objects by ``const&``:

* Prefer ``const raisim::Vec<3>&`` / ``const raisim::Mat<3,3>&`` for fixed-size.
* Prefer ``const raisim::VecDyn&`` / ``const raisim::MatDyn&`` for dynamic-size.
* When accepting either RaiSim or Eigen vectors, use ``Eigen::Ref`` and pass
  ``vec.e()`` from RaiSim (e.g. ``const Eigen::Ref<const Eigen::VectorXd>&``).

Blocks and helpers
=============================
``raisim::Mat`` provides a small Eigen-like API for sub-views:

* ``row(i)``, ``col(j)``
* ``segment<rows, cols>(startRow, startCol)``
* vector-only: ``head<k>()`` / ``tail<k>()``
* corners: ``topLeftCorner<r,c>()`` / ``bottomRightCorner<r,c>()`` etc.

There are also common vector/matrix helpers implemented directly on expressions:

* ``sum()``, ``norm()``, ``squaredNorm()``, ``dot(other)``
* scalar ops: ``*=`` ``/=`` ``+=`` ``-=``
* 3D vector helpers: ``cross(other)``, ``skew()`` (useful for building skew matrices)

Common pitfalls
=============================
* **Column-major layout:** RaiSim matrices are column-major. When filling from
  row-major arrays, transpose or fill by columns.
* **Dangling maps/pointers:** ``.e()`` and ``ptr()`` become invalid after
  ``resize()`` (dynamic types) or when the object goes out of scope.
* **Size mismatches:** ``VecDyn::operator=(Eigen::VectorXd)`` checks dimensions
  at runtime. Resize first or use ``setZero(size)`` before assignment.
