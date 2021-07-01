import control
import numpy

# x_dot = Ax + Bu
# y = Cx + Du

numpy.set_printoptions(suppress=True, precision=3, linewidth=250)

controllable = False
while not controllable:
    A = numpy.random.uniform(-1, 1, size=(30, 30))
    B = numpy.random.uniform(-2, 2, size=(30, 5))

    C = control.ctrb(A, B)
    length_A = max(A.shape)
    ctrl_rank = numpy.linalg.matrix_rank(C)

    if (length_A - ctrl_rank) <= 0:
        controllable = True
        numpy.savetxt("A_controllable.csv", A, delimiter=",")
        numpy.savetxt("B_controllable.csv", B, delimiter=",")

    print(ctrl_rank)
