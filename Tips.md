# Tips

- File path in `build` folder
- Depth image file type `.pgm`
- new g2o version has some update, https://github.com/gaoxiang12/slambook/pull/85/files, https://www.codeleading.com/article/63435719551/
- In ch9, i don't know why i can't use `.so` type to create target. Use `.a` instead. Change `add_library` into `add_executable` in `src/CMakeLists.txt`. 
- New version Sophus  changes `rotation_matrix()` into `matrix()`.
- The tum dataset encountered a python version problem using `associate.py `, https://blog.csdn.net/u012796629/article/details/87932936
- add glog to target_link_libraries
