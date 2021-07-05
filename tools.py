import control
import numpy

# x_dot = Ax + Bu
# y = Cx + Du

numpy.set_printoptions(suppress=True, precision=3, linewidth=250)


def generate_model():
    controllable = False
    observable = False

    global A, B, C, D

    while not controllable or not observable:
        A = numpy.random.randint(-10, 10, size=(5, 5))
        B = numpy.random.randint(-20, 20, size=(5, 2))
        C = numpy.random.randint(-10, 10, size=(1, 5))
        D = numpy.random.randint(-20, 20, size=(2, 1))

        ctrb = control.ctrb(A, B)
        obsv = control.obsv(A, C)

        length_A = max(A.shape)
        ctrl_rank = numpy.linalg.matrix_rank(ctrb)
        obsv_rank = numpy.linalg.matrix_rank(obsv)

        if (length_A - ctrl_rank) == 0 and (length_A - obsv_rank) == 0:
            controllable = True
            observable = True
            numpy.savetxt("A.csv", A, delimiter=",")
            numpy.savetxt("B.csv", B, delimiter=",")
            numpy.savetxt("C.csv", A, delimiter=",")
            numpy.savetxt("D.csv", B, delimiter=",")

        print(ctrl_rank, obsv_rank)

generate_model()