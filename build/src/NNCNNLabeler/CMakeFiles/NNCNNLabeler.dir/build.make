# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.7

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /data/xzjiang/GPU-study/N3LDGClassifier-change-for-gpu-second-time

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /data/xzjiang/GPU-study/N3LDGClassifier-change-for-gpu-second-time/build

# Include any dependencies generated for this target.
include src/NNCNNLabeler/CMakeFiles/NNCNNLabeler.dir/depend.make

# Include the progress variables for this target.
include src/NNCNNLabeler/CMakeFiles/NNCNNLabeler.dir/progress.make

# Include the compile flags for this target's objects.
include src/NNCNNLabeler/CMakeFiles/NNCNNLabeler.dir/flags.make

src/NNCNNLabeler/CMakeFiles/NNCNNLabeler.dir/NNCNNLabeler.cpp.o: src/NNCNNLabeler/CMakeFiles/NNCNNLabeler.dir/flags.make
src/NNCNNLabeler/CMakeFiles/NNCNNLabeler.dir/NNCNNLabeler.cpp.o: ../src/NNCNNLabeler/NNCNNLabeler.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/data/xzjiang/GPU-study/N3LDGClassifier-change-for-gpu-second-time/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/NNCNNLabeler/CMakeFiles/NNCNNLabeler.dir/NNCNNLabeler.cpp.o"
	cd /data/xzjiang/GPU-study/N3LDGClassifier-change-for-gpu-second-time/build/src/NNCNNLabeler && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/NNCNNLabeler.dir/NNCNNLabeler.cpp.o -c /data/xzjiang/GPU-study/N3LDGClassifier-change-for-gpu-second-time/src/NNCNNLabeler/NNCNNLabeler.cpp

src/NNCNNLabeler/CMakeFiles/NNCNNLabeler.dir/NNCNNLabeler.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/NNCNNLabeler.dir/NNCNNLabeler.cpp.i"
	cd /data/xzjiang/GPU-study/N3LDGClassifier-change-for-gpu-second-time/build/src/NNCNNLabeler && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /data/xzjiang/GPU-study/N3LDGClassifier-change-for-gpu-second-time/src/NNCNNLabeler/NNCNNLabeler.cpp > CMakeFiles/NNCNNLabeler.dir/NNCNNLabeler.cpp.i

src/NNCNNLabeler/CMakeFiles/NNCNNLabeler.dir/NNCNNLabeler.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/NNCNNLabeler.dir/NNCNNLabeler.cpp.s"
	cd /data/xzjiang/GPU-study/N3LDGClassifier-change-for-gpu-second-time/build/src/NNCNNLabeler && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /data/xzjiang/GPU-study/N3LDGClassifier-change-for-gpu-second-time/src/NNCNNLabeler/NNCNNLabeler.cpp -o CMakeFiles/NNCNNLabeler.dir/NNCNNLabeler.cpp.s

src/NNCNNLabeler/CMakeFiles/NNCNNLabeler.dir/NNCNNLabeler.cpp.o.requires:

.PHONY : src/NNCNNLabeler/CMakeFiles/NNCNNLabeler.dir/NNCNNLabeler.cpp.o.requires

src/NNCNNLabeler/CMakeFiles/NNCNNLabeler.dir/NNCNNLabeler.cpp.o.provides: src/NNCNNLabeler/CMakeFiles/NNCNNLabeler.dir/NNCNNLabeler.cpp.o.requires
	$(MAKE) -f src/NNCNNLabeler/CMakeFiles/NNCNNLabeler.dir/build.make src/NNCNNLabeler/CMakeFiles/NNCNNLabeler.dir/NNCNNLabeler.cpp.o.provides.build
.PHONY : src/NNCNNLabeler/CMakeFiles/NNCNNLabeler.dir/NNCNNLabeler.cpp.o.provides

src/NNCNNLabeler/CMakeFiles/NNCNNLabeler.dir/NNCNNLabeler.cpp.o.provides.build: src/NNCNNLabeler/CMakeFiles/NNCNNLabeler.dir/NNCNNLabeler.cpp.o


# Object files for target NNCNNLabeler
NNCNNLabeler_OBJECTS = \
"CMakeFiles/NNCNNLabeler.dir/NNCNNLabeler.cpp.o"

# External object files for target NNCNNLabeler
NNCNNLabeler_EXTERNAL_OBJECTS =

../NNCNNLabeler: src/NNCNNLabeler/CMakeFiles/NNCNNLabeler.dir/NNCNNLabeler.cpp.o
../NNCNNLabeler: src/NNCNNLabeler/CMakeFiles/NNCNNLabeler.dir/build.make
../NNCNNLabeler: matrix/libmatrix.a
../NNCNNLabeler: src/NNCNNLabeler/CMakeFiles/NNCNNLabeler.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/data/xzjiang/GPU-study/N3LDGClassifier-change-for-gpu-second-time/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../../NNCNNLabeler"
	cd /data/xzjiang/GPU-study/N3LDGClassifier-change-for-gpu-second-time/build/src/NNCNNLabeler && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/NNCNNLabeler.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/NNCNNLabeler/CMakeFiles/NNCNNLabeler.dir/build: ../NNCNNLabeler

.PHONY : src/NNCNNLabeler/CMakeFiles/NNCNNLabeler.dir/build

src/NNCNNLabeler/CMakeFiles/NNCNNLabeler.dir/requires: src/NNCNNLabeler/CMakeFiles/NNCNNLabeler.dir/NNCNNLabeler.cpp.o.requires

.PHONY : src/NNCNNLabeler/CMakeFiles/NNCNNLabeler.dir/requires

src/NNCNNLabeler/CMakeFiles/NNCNNLabeler.dir/clean:
	cd /data/xzjiang/GPU-study/N3LDGClassifier-change-for-gpu-second-time/build/src/NNCNNLabeler && $(CMAKE_COMMAND) -P CMakeFiles/NNCNNLabeler.dir/cmake_clean.cmake
.PHONY : src/NNCNNLabeler/CMakeFiles/NNCNNLabeler.dir/clean

src/NNCNNLabeler/CMakeFiles/NNCNNLabeler.dir/depend:
	cd /data/xzjiang/GPU-study/N3LDGClassifier-change-for-gpu-second-time/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /data/xzjiang/GPU-study/N3LDGClassifier-change-for-gpu-second-time /data/xzjiang/GPU-study/N3LDGClassifier-change-for-gpu-second-time/src/NNCNNLabeler /data/xzjiang/GPU-study/N3LDGClassifier-change-for-gpu-second-time/build /data/xzjiang/GPU-study/N3LDGClassifier-change-for-gpu-second-time/build/src/NNCNNLabeler /data/xzjiang/GPU-study/N3LDGClassifier-change-for-gpu-second-time/build/src/NNCNNLabeler/CMakeFiles/NNCNNLabeler.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/NNCNNLabeler/CMakeFiles/NNCNNLabeler.dir/depend

