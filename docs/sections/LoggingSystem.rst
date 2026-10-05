#############################
Logging System
#############################

The RaiSim logging system, inspired by the ``glog`` library, provides one-line
macros (``raisim/raisim_message.hpp``, included by ``raisim/World.hpp``) that
print a message together with a time stamp and its source location, in the form
``[YYYY:MM:DD:HH:MM:SS file.cpp:42] message``.
Messages are written only to standard output (``std::cout``); they are not saved to a file.
The message argument can be anything that can be streamed with ``<<``, e.g.
``RSWARN("mass is " << mass)``.
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

- ``RSINFO(msg)``: Prints the message in the default terminal color.
- ``RSWARN(msg)``: Prints the message in yellow.
- ``RSFATAL(msg)``: Prints the message in bold red and calls the fatal callback.
- ``RSINFO_IF(con, msg)``: Prints the message in the default color if the condition evaluates to true.
- ``RSWARN_IF(con, msg)``: Prints the message in yellow if the condition evaluates to true.
- ``RSFATAL_IF(con, msg)``: Prints the message in bold red and calls the fatal callback if the condition evaluates to true.
- ``RSASSERT(con, msg)``: Prints the message in bold red and calls the fatal callback if the condition evaluates to false. It is active in all builds.
- ``RSRETURN_IF(con, msg)``: Prints the message in the default color and executes ``return;`` if the condition evaluates to true. Use it only in functions that return ``void``.
- ``RSISNAN(val)``: Prints ``val is nan`` (with the expression text in place of ``val``) and calls the fatal callback if the value is NaN (Not a Number).
- ``RSISNAN_MSG(val, msg)``: Prints the specified message and calls the fatal callback if the value is NaN.

``printf``-style variants take a format string and arguments: ``RSINFOF``,
``RSWARNF``, ``RSFATALF``, ``RSINFO_IFF``, ``RSWARN_IFF``, ``RSFATAL_IFF``,
``RSASSERTF``, ``RSRETURN_IFF``, and ``RSISNAN_MSGF``, e.g.
``RSWARNF("mass is %f", mass)``. The formatted message is truncated to 1023
characters.

Fatal callback
==============

By default, a fatal message calls ``std::exit(1)``.
Install a global callback to handle fatal errors differently, for example by
throwing an exception that the application can catch:

.. code-block:: cpp

    raisim::RaiSimMsg::setFatalCallback([]() {
      throw std::runtime_error("RaiSim fatal error; see the log above");
    });

Execution continues after the fatal macro if the callback returns, so a custom
callback should normally throw or terminate. Do not use ``[](){ throw; }``: a
``throw;`` without an exception being handled calls ``std::terminate``.
Passing an empty function restores the default.

``RaiSimMsg::setThreadFatalCallback(callback)`` installs a callback for the
calling thread only; it takes precedence over the global callback on that
thread and returns the thread's previous callback. The RAII helper
``raisim::ScopedRaiSimFatalCallback`` installs a thread callback and restores
the previous one when it goes out of scope. Thread callbacks are kept by thread
id, so remove one (pass an empty function, or let the scoped helper do it)
before its thread exits; a callback left behind would be inherited by a later
thread that is given the same id.

Debug-only macros
=================

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

They expand to the corresponding ``RS*`` macros only when ``RSDEBUG`` is
defined, and to nothing otherwise; ``printf``-style ``DRS*F`` variants exist as
well. The exported ``raisim::raisim`` CMake target defines ``RSDEBUG`` when your
build links the Debug RaiSim library, so a Debug build of your application has
these checks only if the RaiSim package contains the Debug library (see
:doc:`BuildAndTest`).

API
====

RaiSimMsg
*********

.. doxygenclass:: raisim::RaiSimMsg
   :members:

