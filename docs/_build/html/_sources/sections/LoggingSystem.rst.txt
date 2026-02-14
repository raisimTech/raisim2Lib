#############################
Logging System
#############################

The RaiSim logging system, inspired by the ``glog`` library, offers a compact method for representing multi-line log entries.
Messages are streamed exclusively to standard output (``std::cout``) and are not persisted to a file.
For instance, rather than:

.. code-block:: cpp

    if(fail) {
        std::cout<<date<<FILENAME<<LINENUMBER<< ... << "failed here \n";
        exit(0);
    }

one may utilize:

.. code-block:: cpp

    RSFATAL_IF(fail, "failed here")

The available logging macros are as follows:

- ``RSINFO(msg)``: Prints the message in white.
- ``RSWARN(msg)``: Prints the message in yellow.
- ``RSFATAL(msg)``: Prints the message in red and executes the fatal function.
- ``RSINFO_IF(con, msg)``: Prints the message in white if the condition evaluates to true.
- ``RSWARN_IF(con, msg)``: Prints the message in yellow if the condition evaluates to true.
- ``RSFATAL_IF(con, msg)``: Prints the message in red and invokes the fatal function if the condition evaluates to true.
- ``RSASSERT(con, msg)``: Prints the message in red and invokes the fatal function if the condition evaluates to false.
- ``RSRETURN_IF(con, msg)``: Prints the message in white and executes a ``return`` statement if the condition evaluates to true.
- ``RSISNAN(val)``: Prints "${val} is NaN" and invokes the fatal function if the value is NaN (Not a Number).
- ``RSISNAN_MSG(val, msg)``: Prints the specified message and invokes the fatal function if the value is NaN.

The default fatal function is ``std::exit(1);``.
This behavior can be overridden as follows:

.. code-block:: cpp

    raisim::RaiSimMsg::setFatalCallback([](){throw;});

These macros impact performance; for example, all ``_IF`` macros involve boolean checks and potential branching.

To avoid performance overhead from branching in release builds, the following debug-only versions of the macros are available:

- ``DRSINFO(msg)``
- ``DRSWARN(msg)``
- ``DRSFATAL(msg)``
- ``DRSINFO_IF(con, msg)``
- ``DRSWARN_IF(con, msg)``
- ``DRSFATAL_IF(con, msg)``
- ``DRSASSERT(con, msg)``
- ``DRSRETURN_IF(con, msg)``
- ``DRSISNAN(val)``
- ``DRSISNAN_MSG(val, msg)``

These macros are compiled out unless the build type is set to Debug.

API
====

RaiSimMsg
*********

.. doxygenclass:: raisim::RaiSimMsg
   :members:

