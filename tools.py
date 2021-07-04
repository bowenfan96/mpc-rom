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
        A = numpy.random.uniform(-1, 1, size=(30, 30))
        B = numpy.random.uniform(-2, 2, size=(30, 5))
        C = numpy.random.uniform(-1, 1, size=(5, 30))
        D = numpy.random.uniform(-2, 2, size=(5, 5))

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

system = control.ss(A, B, C, D)
rom = control.balred(system, 10, method='truncate')

print(rom)
