# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/spurs/app/cmake-3.22.2-linux-x86_64/bin/cmake

# The command to remove a file.
RM = /home/spurs/app/cmake-3.22.2-linux-x86_64/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/spurs/x/VINS-Fusion/ThirdParty/DBow

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/spurs/x/VINS-Fusion/ThirdParty/DBow/build

# Include any dependencies generated for this target.
include CMakeFiles/DBoW2.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/DBoW2.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/DBoW2.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/DBoW2.dir/flags.make

CMakeFiles/DBoW2.dir/DBoW/BowVector.cpp.o: CMakeFiles/DBoW2.dir/flags.make
CMakeFiles/DBoW2.dir/DBoW/BowVector.cpp.o: ../DBoW/BowVector.cpp
CMakeFiles/DBoW2.dir/DBoW/BowVector.cpp.o: CMakeFiles/DBoW2.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/spurs/x/VINS-Fusion/ThirdParty/DBow/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/DBoW2.dir/DBoW/BowVector.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/DBoW2.dir/DBoW/BowVector.cpp.o -MF CMakeFiles/DBoW2.dir/DBoW/BowVector.cpp.o.d -o CMakeFiles/DBoW2.dir/DBoW/BowVector.cpp.o -c /home/spurs/x/VINS-Fusion/ThirdParty/DBow/DBoW/BowVector.cpp

CMakeFiles/DBoW2.dir/DBoW/BowVector.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/DBoW2.dir/DBoW/BowVector.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/spurs/x/VINS-Fusion/ThirdParty/DBow/DBoW/BowVector.cpp > CMakeFiles/DBoW2.dir/DBoW/BowVector.cpp.i

CMakeFiles/DBoW2.dir/DBoW/BowVector.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/DBoW2.dir/DBoW/BowVector.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/spurs/x/VINS-Fusion/ThirdParty/DBow/DBoW/BowVector.cpp -o CMakeFiles/DBoW2.dir/DBoW/BowVector.cpp.s

CMakeFiles/DBoW2.dir/DBoW/FBrief.cpp.o: CMakeFiles/DBoW2.dir/flags.make
CMakeFiles/DBoW2.dir/DBoW/FBrief.cpp.o: ../DBoW/FBrief.cpp
CMakeFiles/DBoW2.dir/DBoW/FBrief.cpp.o: CMakeFiles/DBoW2.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/spurs/x/VINS-Fusion/ThirdParty/DBow/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/DBoW2.dir/DBoW/FBrief.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/DBoW2.dir/DBoW/FBrief.cpp.o -MF CMakeFiles/DBoW2.dir/DBoW/FBrief.cpp.o.d -o CMakeFiles/DBoW2.dir/DBoW/FBrief.cpp.o -c /home/spurs/x/VINS-Fusion/ThirdParty/DBow/DBoW/FBrief.cpp

CMakeFiles/DBoW2.dir/DBoW/FBrief.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/DBoW2.dir/DBoW/FBrief.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/spurs/x/VINS-Fusion/ThirdParty/DBow/DBoW/FBrief.cpp > CMakeFiles/DBoW2.dir/DBoW/FBrief.cpp.i

CMakeFiles/DBoW2.dir/DBoW/FBrief.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/DBoW2.dir/DBoW/FBrief.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/spurs/x/VINS-Fusion/ThirdParty/DBow/DBoW/FBrief.cpp -o CMakeFiles/DBoW2.dir/DBoW/FBrief.cpp.s

CMakeFiles/DBoW2.dir/DBoW/FeatureVector.cpp.o: CMakeFiles/DBoW2.dir/flags.make
CMakeFiles/DBoW2.dir/DBoW/FeatureVector.cpp.o: ../DBoW/FeatureVector.cpp
CMakeFiles/DBoW2.dir/DBoW/FeatureVector.cpp.o: CMakeFiles/DBoW2.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/spurs/x/VINS-Fusion/ThirdParty/DBow/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/DBoW2.dir/DBoW/FeatureVector.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/DBoW2.dir/DBoW/FeatureVector.cpp.o -MF CMakeFiles/DBoW2.dir/DBoW/FeatureVector.cpp.o.d -o CMakeFiles/DBoW2.dir/DBoW/FeatureVector.cpp.o -c /home/spurs/x/VINS-Fusion/ThirdParty/DBow/DBoW/FeatureVector.cpp

CMakeFiles/DBoW2.dir/DBoW/FeatureVector.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/DBoW2.dir/DBoW/FeatureVector.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/spurs/x/VINS-Fusion/ThirdParty/DBow/DBoW/FeatureVector.cpp > CMakeFiles/DBoW2.dir/DBoW/FeatureVector.cpp.i

CMakeFiles/DBoW2.dir/DBoW/FeatureVector.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/DBoW2.dir/DBoW/FeatureVector.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/spurs/x/VINS-Fusion/ThirdParty/DBow/DBoW/FeatureVector.cpp -o CMakeFiles/DBoW2.dir/DBoW/FeatureVector.cpp.s

CMakeFiles/DBoW2.dir/DBoW/QueryResults.cpp.o: CMakeFiles/DBoW2.dir/flags.make
CMakeFiles/DBoW2.dir/DBoW/QueryResults.cpp.o: ../DBoW/QueryResults.cpp
CMakeFiles/DBoW2.dir/DBoW/QueryResults.cpp.o: CMakeFiles/DBoW2.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/spurs/x/VINS-Fusion/ThirdParty/DBow/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/DBoW2.dir/DBoW/QueryResults.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/DBoW2.dir/DBoW/QueryResults.cpp.o -MF CMakeFiles/DBoW2.dir/DBoW/QueryResults.cpp.o.d -o CMakeFiles/DBoW2.dir/DBoW/QueryResults.cpp.o -c /home/spurs/x/VINS-Fusion/ThirdParty/DBow/DBoW/QueryResults.cpp

CMakeFiles/DBoW2.dir/DBoW/QueryResults.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/DBoW2.dir/DBoW/QueryResults.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/spurs/x/VINS-Fusion/ThirdParty/DBow/DBoW/QueryResults.cpp > CMakeFiles/DBoW2.dir/DBoW/QueryResults.cpp.i

CMakeFiles/DBoW2.dir/DBoW/QueryResults.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/DBoW2.dir/DBoW/QueryResults.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/spurs/x/VINS-Fusion/ThirdParty/DBow/DBoW/QueryResults.cpp -o CMakeFiles/DBoW2.dir/DBoW/QueryResults.cpp.s

CMakeFiles/DBoW2.dir/DBoW/ScoringObject.cpp.o: CMakeFiles/DBoW2.dir/flags.make
CMakeFiles/DBoW2.dir/DBoW/ScoringObject.cpp.o: ../DBoW/ScoringObject.cpp
CMakeFiles/DBoW2.dir/DBoW/ScoringObject.cpp.o: CMakeFiles/DBoW2.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/spurs/x/VINS-Fusion/ThirdParty/DBow/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/DBoW2.dir/DBoW/ScoringObject.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/DBoW2.dir/DBoW/ScoringObject.cpp.o -MF CMakeFiles/DBoW2.dir/DBoW/ScoringObject.cpp.o.d -o CMakeFiles/DBoW2.dir/DBoW/ScoringObject.cpp.o -c /home/spurs/x/VINS-Fusion/ThirdParty/DBow/DBoW/ScoringObject.cpp

CMakeFiles/DBoW2.dir/DBoW/ScoringObject.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/DBoW2.dir/DBoW/ScoringObject.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/spurs/x/VINS-Fusion/ThirdParty/DBow/DBoW/ScoringObject.cpp > CMakeFiles/DBoW2.dir/DBoW/ScoringObject.cpp.i

CMakeFiles/DBoW2.dir/DBoW/ScoringObject.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/DBoW2.dir/DBoW/ScoringObject.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/spurs/x/VINS-Fusion/ThirdParty/DBow/DBoW/ScoringObject.cpp -o CMakeFiles/DBoW2.dir/DBoW/ScoringObject.cpp.s

CMakeFiles/DBoW2.dir/DUtils/Random.cpp.o: CMakeFiles/DBoW2.dir/flags.make
CMakeFiles/DBoW2.dir/DUtils/Random.cpp.o: ../DUtils/Random.cpp
CMakeFiles/DBoW2.dir/DUtils/Random.cpp.o: CMakeFiles/DBoW2.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/spurs/x/VINS-Fusion/ThirdParty/DBow/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/DBoW2.dir/DUtils/Random.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/DBoW2.dir/DUtils/Random.cpp.o -MF CMakeFiles/DBoW2.dir/DUtils/Random.cpp.o.d -o CMakeFiles/DBoW2.dir/DUtils/Random.cpp.o -c /home/spurs/x/VINS-Fusion/ThirdParty/DBow/DUtils/Random.cpp

CMakeFiles/DBoW2.dir/DUtils/Random.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/DBoW2.dir/DUtils/Random.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/spurs/x/VINS-Fusion/ThirdParty/DBow/DUtils/Random.cpp > CMakeFiles/DBoW2.dir/DUtils/Random.cpp.i

CMakeFiles/DBoW2.dir/DUtils/Random.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/DBoW2.dir/DUtils/Random.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/spurs/x/VINS-Fusion/ThirdParty/DBow/DUtils/Random.cpp -o CMakeFiles/DBoW2.dir/DUtils/Random.cpp.s

CMakeFiles/DBoW2.dir/DUtils/Timestamp.cpp.o: CMakeFiles/DBoW2.dir/flags.make
CMakeFiles/DBoW2.dir/DUtils/Timestamp.cpp.o: ../DUtils/Timestamp.cpp
CMakeFiles/DBoW2.dir/DUtils/Timestamp.cpp.o: CMakeFiles/DBoW2.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/spurs/x/VINS-Fusion/ThirdParty/DBow/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object CMakeFiles/DBoW2.dir/DUtils/Timestamp.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/DBoW2.dir/DUtils/Timestamp.cpp.o -MF CMakeFiles/DBoW2.dir/DUtils/Timestamp.cpp.o.d -o CMakeFiles/DBoW2.dir/DUtils/Timestamp.cpp.o -c /home/spurs/x/VINS-Fusion/ThirdParty/DBow/DUtils/Timestamp.cpp

CMakeFiles/DBoW2.dir/DUtils/Timestamp.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/DBoW2.dir/DUtils/Timestamp.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/spurs/x/VINS-Fusion/ThirdParty/DBow/DUtils/Timestamp.cpp > CMakeFiles/DBoW2.dir/DUtils/Timestamp.cpp.i

CMakeFiles/DBoW2.dir/DUtils/Timestamp.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/DBoW2.dir/DUtils/Timestamp.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/spurs/x/VINS-Fusion/ThirdParty/DBow/DUtils/Timestamp.cpp -o CMakeFiles/DBoW2.dir/DUtils/Timestamp.cpp.s

CMakeFiles/DBoW2.dir/DVision/BRIEF.cpp.o: CMakeFiles/DBoW2.dir/flags.make
CMakeFiles/DBoW2.dir/DVision/BRIEF.cpp.o: ../DVision/BRIEF.cpp
CMakeFiles/DBoW2.dir/DVision/BRIEF.cpp.o: CMakeFiles/DBoW2.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/spurs/x/VINS-Fusion/ThirdParty/DBow/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object CMakeFiles/DBoW2.dir/DVision/BRIEF.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/DBoW2.dir/DVision/BRIEF.cpp.o -MF CMakeFiles/DBoW2.dir/DVision/BRIEF.cpp.o.d -o CMakeFiles/DBoW2.dir/DVision/BRIEF.cpp.o -c /home/spurs/x/VINS-Fusion/ThirdParty/DBow/DVision/BRIEF.cpp

CMakeFiles/DBoW2.dir/DVision/BRIEF.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/DBoW2.dir/DVision/BRIEF.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/spurs/x/VINS-Fusion/ThirdParty/DBow/DVision/BRIEF.cpp > CMakeFiles/DBoW2.dir/DVision/BRIEF.cpp.i

CMakeFiles/DBoW2.dir/DVision/BRIEF.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/DBoW2.dir/DVision/BRIEF.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/spurs/x/VINS-Fusion/ThirdParty/DBow/DVision/BRIEF.cpp -o CMakeFiles/DBoW2.dir/DVision/BRIEF.cpp.s

CMakeFiles/DBoW2.dir/DBoW/VocabularyBinary.cpp.o: CMakeFiles/DBoW2.dir/flags.make
CMakeFiles/DBoW2.dir/DBoW/VocabularyBinary.cpp.o: ../DBoW/VocabularyBinary.cpp
CMakeFiles/DBoW2.dir/DBoW/VocabularyBinary.cpp.o: CMakeFiles/DBoW2.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/spurs/x/VINS-Fusion/ThirdParty/DBow/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object CMakeFiles/DBoW2.dir/DBoW/VocabularyBinary.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/DBoW2.dir/DBoW/VocabularyBinary.cpp.o -MF CMakeFiles/DBoW2.dir/DBoW/VocabularyBinary.cpp.o.d -o CMakeFiles/DBoW2.dir/DBoW/VocabularyBinary.cpp.o -c /home/spurs/x/VINS-Fusion/ThirdParty/DBow/DBoW/VocabularyBinary.cpp

CMakeFiles/DBoW2.dir/DBoW/VocabularyBinary.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/DBoW2.dir/DBoW/VocabularyBinary.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/spurs/x/VINS-Fusion/ThirdParty/DBow/DBoW/VocabularyBinary.cpp > CMakeFiles/DBoW2.dir/DBoW/VocabularyBinary.cpp.i

CMakeFiles/DBoW2.dir/DBoW/VocabularyBinary.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/DBoW2.dir/DBoW/VocabularyBinary.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/spurs/x/VINS-Fusion/ThirdParty/DBow/DBoW/VocabularyBinary.cpp -o CMakeFiles/DBoW2.dir/DBoW/VocabularyBinary.cpp.s

# Object files for target DBoW2
DBoW2_OBJECTS = \
"CMakeFiles/DBoW2.dir/DBoW/BowVector.cpp.o" \
"CMakeFiles/DBoW2.dir/DBoW/FBrief.cpp.o" \
"CMakeFiles/DBoW2.dir/DBoW/FeatureVector.cpp.o" \
"CMakeFiles/DBoW2.dir/DBoW/QueryResults.cpp.o" \
"CMakeFiles/DBoW2.dir/DBoW/ScoringObject.cpp.o" \
"CMakeFiles/DBoW2.dir/DUtils/Random.cpp.o" \
"CMakeFiles/DBoW2.dir/DUtils/Timestamp.cpp.o" \
"CMakeFiles/DBoW2.dir/DVision/BRIEF.cpp.o" \
"CMakeFiles/DBoW2.dir/DBoW/VocabularyBinary.cpp.o"

# External object files for target DBoW2
DBoW2_EXTERNAL_OBJECTS =

../lib/libDBoW2.so: CMakeFiles/DBoW2.dir/DBoW/BowVector.cpp.o
../lib/libDBoW2.so: CMakeFiles/DBoW2.dir/DBoW/FBrief.cpp.o
../lib/libDBoW2.so: CMakeFiles/DBoW2.dir/DBoW/FeatureVector.cpp.o
../lib/libDBoW2.so: CMakeFiles/DBoW2.dir/DBoW/QueryResults.cpp.o
../lib/libDBoW2.so: CMakeFiles/DBoW2.dir/DBoW/ScoringObject.cpp.o
../lib/libDBoW2.so: CMakeFiles/DBoW2.dir/DUtils/Random.cpp.o
../lib/libDBoW2.so: CMakeFiles/DBoW2.dir/DUtils/Timestamp.cpp.o
../lib/libDBoW2.so: CMakeFiles/DBoW2.dir/DVision/BRIEF.cpp.o
../lib/libDBoW2.so: CMakeFiles/DBoW2.dir/DBoW/VocabularyBinary.cpp.o
../lib/libDBoW2.so: CMakeFiles/DBoW2.dir/build.make
../lib/libDBoW2.so: /home/spurs/installed/lib/libopencv_gapi.so.4.5.0
../lib/libDBoW2.so: /home/spurs/installed/lib/libopencv_stitching.so.4.5.0
../lib/libDBoW2.so: /home/spurs/installed/lib/libopencv_alphamat.so.4.5.0
../lib/libDBoW2.so: /home/spurs/installed/lib/libopencv_aruco.so.4.5.0
../lib/libDBoW2.so: /home/spurs/installed/lib/libopencv_bgsegm.so.4.5.0
../lib/libDBoW2.so: /home/spurs/installed/lib/libopencv_bioinspired.so.4.5.0
../lib/libDBoW2.so: /home/spurs/installed/lib/libopencv_ccalib.so.4.5.0
../lib/libDBoW2.so: /home/spurs/installed/lib/libopencv_dnn_objdetect.so.4.5.0
../lib/libDBoW2.so: /home/spurs/installed/lib/libopencv_dnn_superres.so.4.5.0
../lib/libDBoW2.so: /home/spurs/installed/lib/libopencv_dpm.so.4.5.0
../lib/libDBoW2.so: /home/spurs/installed/lib/libopencv_face.so.4.5.0
../lib/libDBoW2.so: /home/spurs/installed/lib/libopencv_freetype.so.4.5.0
../lib/libDBoW2.so: /home/spurs/installed/lib/libopencv_fuzzy.so.4.5.0
../lib/libDBoW2.so: /home/spurs/installed/lib/libopencv_hdf.so.4.5.0
../lib/libDBoW2.so: /home/spurs/installed/lib/libopencv_hfs.so.4.5.0
../lib/libDBoW2.so: /home/spurs/installed/lib/libopencv_img_hash.so.4.5.0
../lib/libDBoW2.so: /home/spurs/installed/lib/libopencv_intensity_transform.so.4.5.0
../lib/libDBoW2.so: /home/spurs/installed/lib/libopencv_line_descriptor.so.4.5.0
../lib/libDBoW2.so: /home/spurs/installed/lib/libopencv_mcc.so.4.5.0
../lib/libDBoW2.so: /home/spurs/installed/lib/libopencv_quality.so.4.5.0
../lib/libDBoW2.so: /home/spurs/installed/lib/libopencv_rapid.so.4.5.0
../lib/libDBoW2.so: /home/spurs/installed/lib/libopencv_reg.so.4.5.0
../lib/libDBoW2.so: /home/spurs/installed/lib/libopencv_rgbd.so.4.5.0
../lib/libDBoW2.so: /home/spurs/installed/lib/libopencv_saliency.so.4.5.0
../lib/libDBoW2.so: /home/spurs/installed/lib/libopencv_sfm.so.4.5.0
../lib/libDBoW2.so: /home/spurs/installed/lib/libopencv_stereo.so.4.5.0
../lib/libDBoW2.so: /home/spurs/installed/lib/libopencv_structured_light.so.4.5.0
../lib/libDBoW2.so: /home/spurs/installed/lib/libopencv_superres.so.4.5.0
../lib/libDBoW2.so: /home/spurs/installed/lib/libopencv_surface_matching.so.4.5.0
../lib/libDBoW2.so: /home/spurs/installed/lib/libopencv_tracking.so.4.5.0
../lib/libDBoW2.so: /home/spurs/installed/lib/libopencv_videostab.so.4.5.0
../lib/libDBoW2.so: /home/spurs/installed/lib/libopencv_viz.so.4.5.0
../lib/libDBoW2.so: /home/spurs/installed/lib/libopencv_xfeatures2d.so.4.5.0
../lib/libDBoW2.so: /home/spurs/installed/lib/libopencv_xobjdetect.so.4.5.0
../lib/libDBoW2.so: /home/spurs/installed/lib/libopencv_xphoto.so.4.5.0
../lib/libDBoW2.so: /home/spurs/installed/lib/libopencv_highgui.so.4.5.0
../lib/libDBoW2.so: /home/spurs/installed/lib/libopencv_shape.so.4.5.0
../lib/libDBoW2.so: /home/spurs/installed/lib/libopencv_datasets.so.4.5.0
../lib/libDBoW2.so: /home/spurs/installed/lib/libopencv_plot.so.4.5.0
../lib/libDBoW2.so: /home/spurs/installed/lib/libopencv_text.so.4.5.0
../lib/libDBoW2.so: /home/spurs/installed/lib/libopencv_dnn.so.4.5.0
../lib/libDBoW2.so: /home/spurs/installed/lib/libopencv_ml.so.4.5.0
../lib/libDBoW2.so: /home/spurs/installed/lib/libopencv_phase_unwrapping.so.4.5.0
../lib/libDBoW2.so: /home/spurs/installed/lib/libopencv_optflow.so.4.5.0
../lib/libDBoW2.so: /home/spurs/installed/lib/libopencv_ximgproc.so.4.5.0
../lib/libDBoW2.so: /home/spurs/installed/lib/libopencv_video.so.4.5.0
../lib/libDBoW2.so: /home/spurs/installed/lib/libopencv_videoio.so.4.5.0
../lib/libDBoW2.so: /home/spurs/installed/lib/libopencv_imgcodecs.so.4.5.0
../lib/libDBoW2.so: /home/spurs/installed/lib/libopencv_objdetect.so.4.5.0
../lib/libDBoW2.so: /home/spurs/installed/lib/libopencv_calib3d.so.4.5.0
../lib/libDBoW2.so: /home/spurs/installed/lib/libopencv_features2d.so.4.5.0
../lib/libDBoW2.so: /home/spurs/installed/lib/libopencv_flann.so.4.5.0
../lib/libDBoW2.so: /home/spurs/installed/lib/libopencv_photo.so.4.5.0
../lib/libDBoW2.so: /home/spurs/installed/lib/libopencv_imgproc.so.4.5.0
../lib/libDBoW2.so: /home/spurs/installed/lib/libopencv_core.so.4.5.0
../lib/libDBoW2.so: CMakeFiles/DBoW2.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/spurs/x/VINS-Fusion/ThirdParty/DBow/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Linking CXX shared library ../lib/libDBoW2.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/DBoW2.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/DBoW2.dir/build: ../lib/libDBoW2.so
.PHONY : CMakeFiles/DBoW2.dir/build

CMakeFiles/DBoW2.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/DBoW2.dir/cmake_clean.cmake
.PHONY : CMakeFiles/DBoW2.dir/clean

CMakeFiles/DBoW2.dir/depend:
	cd /home/spurs/x/VINS-Fusion/ThirdParty/DBow/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/spurs/x/VINS-Fusion/ThirdParty/DBow /home/spurs/x/VINS-Fusion/ThirdParty/DBow /home/spurs/x/VINS-Fusion/ThirdParty/DBow/build /home/spurs/x/VINS-Fusion/ThirdParty/DBow/build /home/spurs/x/VINS-Fusion/ThirdParty/DBow/build/CMakeFiles/DBoW2.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/DBoW2.dir/depend

