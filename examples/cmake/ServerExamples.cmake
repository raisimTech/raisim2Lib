# Facts about the RaiSim examples that serve a world to viewers with RaisimServer, and about the
# rayrai examples that draw through ExampleApp. The native examples build lists the server examples
# next to the executables (raisim_server_examples.tsv), and the browser build compiles each kind
# from the same sources into one WebAssembly module.

# The title of example app_name: the heading of its documentation page, or its name.
function(_raisim_example_title examples_dir app_name out_title)
    string(REPLACE "_" " " _title "${app_name}")
    string(SUBSTRING "${_title}" 0 1 _first)
    string(SUBSTRING "${_title}" 1 -1 _rest)
    string(TOUPPER "${_first}" _first)
    set(_title "${_first}${_rest}")
    file(GLOB _pages "${examples_dir}/../docs/sections/examples/*/${app_name}.md")
    list(FILTER _pages EXCLUDE REGEX "/current/")
    if(_pages)
        list(GET _pages 0 _page)
        file(STRINGS "${_page}" _heading LIMIT_COUNT 1 REGEX "^# ")
        string(REGEX REPLACE "^# ([^:]*: )?" "" _title "${_heading}")
    endif()
    string(REPLACE "\t" " " _title "${_title}")
    set(${out_title} "${_title}" PARENT_SCOPE)
endfunction()

# The signature of the main in source: "plain" or "arguments".
function(_raisim_main_signature source out_signature)
    if(source MATCHES "int[ \t]+main[ \t]*\\([ \t]*\\)")
        set(${out_signature} plain PARENT_SCOPE)
    else()
        set(${out_signature} arguments PARENT_SCOPE)
    endif()
endfunction()

# Sets out_server to whether file_name (relative to examples_dir) is a RaisimServer example, and
# for one its category (the folder under src/server, or under src), its title (the heading of its
# documentation page, or its name) and the signature of its main: "plain" or "arguments".
function(raisim_server_example_facts examples_dir app_name file_name out_server out_category out_title out_signature)
    set(${out_server} FALSE PARENT_SCOPE)
    if(NOT file_name MATCHES "^src/")
        return()
    endif()
    file(READ "${examples_dir}/${file_name}" _source)
    if(NOT _source MATCHES "#include[ \t]*[<\"]raisim/RaisimServer\\.hpp[>\"]")
        return()
    endif()
    string(REGEX MATCH "^src/([a-z_]+)/(([a-z_]+)/)?" _path "${file_name}")
    if(CMAKE_MATCH_3 AND CMAKE_MATCH_1 STREQUAL "server")
        set(_category "${CMAKE_MATCH_3}")
    else()
        set(_category "${CMAKE_MATCH_1}")
    endif()
    _raisim_example_title("${examples_dir}" ${app_name} _title)
    _raisim_main_signature("${_source}" _signature)
    set(${out_server} TRUE PARENT_SCOPE)
    set(${out_category} "${_category}" PARENT_SCOPE)
    set(${out_title} "${_title}" PARENT_SCOPE)
    set(${out_signature} "${_signature}" PARENT_SCOPE)
endfunction()

# Sets out_rayrai to whether file_name (relative to examples_dir) is a rayrai example a browser
# runs: a documented one under src/rayrai that draws through ExampleApp (rayrai/example_common.hpp),
# so its documentation page is where it starts. For one,
# out_name is its file name without extension (its target name), out_category the folder under
# src/rayrai, out_title its title and out_signature the signature of its main.
function(raisim_rayrai_example_facts examples_dir file_name out_rayrai out_name out_category out_title out_signature)
    set(${out_rayrai} FALSE PARENT_SCOPE)
    if(NOT file_name MATCHES "^src/rayrai/([a-z_]+)/([a-z_0-9]+)\\.cpp$")
        return()
    endif()
    set(_category "${CMAKE_MATCH_1}")
    set(_name "${CMAKE_MATCH_2}")
    if(NOT EXISTS "${examples_dir}/../docs/sections/examples/rayrai/${_name}.md")
        return()
    endif()
    file(READ "${examples_dir}/${file_name}" _source)
    # ExampleApp may come through a header next to the source (forest_viewer.hpp).
    set(_texts "${_source}")
    get_filename_component(_folder "${examples_dir}/${file_name}" DIRECTORY)
    string(REGEX MATCHALL "#include[ \t]*\"[a-z_0-9]+\\.hpp\"" _includes "${_source}")
    foreach(_include IN LISTS _includes)
        string(REGEX REPLACE ".*\"([a-z_0-9]+\\.hpp)\"" "\\1" _header "${_include}")
        if(EXISTS "${_folder}/${_header}")
            file(READ "${_folder}/${_header}" _text)
            string(APPEND _texts "${_text}")
        endif()
    endforeach()
    if(NOT _texts MATCHES "#include[ \t]*[<\"]rayrai/example_common\\.hpp[>\"]" OR
       _texts MATCHES "#include[ \t]*[<\"]raisim/RaisimServer\\.hpp[>\"]")
        return()
    endif()
    _raisim_example_title("${examples_dir}" ${_name} _title)
    _raisim_main_signature("${_source}" _signature)
    set(${out_rayrai} TRUE PARENT_SCOPE)
    set(${out_name} "${_name}" PARENT_SCOPE)
    set(${out_category} "${_category}" PARENT_SCOPE)
    set(${out_title} "${_title}" PARENT_SCOPE)
    set(${out_signature} "${_signature}" PARENT_SCOPE)
endfunction()

# The examples examples_dir/CMakeLists.txt builds with create_executable(): out_names and
# out_sources list their target names and source files.
function(raisim_list_examples examples_dir out_names out_sources)
    file(STRINGS "${examples_dir}/CMakeLists.txt" _calls REGEX "create_(rayrai_)?executable\\([a-z_0-9]+ [^ )]+")
    set(_names)
    set(_sources)
    foreach(_call IN LISTS _calls)
        string(REGEX MATCH "create_(rayrai_)?executable\\(([a-z_0-9]+) ([^ )]+)" _match "${_call}")
        if(NOT CMAKE_MATCH_2 IN_LIST _names)
            list(APPEND _names "${CMAKE_MATCH_2}")
            list(APPEND _sources "${CMAKE_MATCH_3}")
        endif()
    endforeach()
    set(${out_names} "${_names}" PARENT_SCOPE)
    set(${out_sources} "${_sources}" PARENT_SCOPE)
endfunction()
