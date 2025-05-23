#!/bin/sh

if [ -n "$DESTDIR" ] ; then
    case $DESTDIR in
        /*) # ok
            ;;
        *)
            /bin/echo "DESTDIR argument must be absolute... "
            /bin/echo "otherwise python's distutils will bork things."
            exit 1
    esac
fi

echo_and_run() { echo "+ $@" ; "$@" ; }

echo_and_run cd "/home/dinisas/SA/catkin_ws/src/pioneer_fast_slam"

# ensure that Python install destination exists
echo_and_run mkdir -p "$DESTDIR/home/dinisas/SA/catkin_ws/install/lib/python3/dist-packages"

# Note that PYTHONPATH is pulled from the environment to support installing
# into one location when some dependencies were installed in another
# location, #123.
echo_and_run /usr/bin/env \
    PYTHONPATH="/home/dinisas/SA/catkin_ws/install/lib/python3/dist-packages:/home/dinisas/SA/catkin_ws/build/lib/python3/dist-packages:$PYTHONPATH" \
    CATKIN_BINARY_DIR="/home/dinisas/SA/catkin_ws/build" \
    "/usr/bin/python3" \
    "/home/dinisas/SA/catkin_ws/src/pioneer_fast_slam/setup.py" \
    egg_info --egg-base /home/dinisas/SA/catkin_ws/build/pioneer_fast_slam \
    build --build-base "/home/dinisas/SA/catkin_ws/build/pioneer_fast_slam" \
    install \
    --root="${DESTDIR-/}" \
    --install-layout=deb --prefix="/home/dinisas/SA/catkin_ws/install" --install-scripts="/home/dinisas/SA/catkin_ws/install/bin"
