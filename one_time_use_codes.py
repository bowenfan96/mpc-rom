# # ----- 3D PLOTTER -----
#
# ctg_plot = np.array(ctg_plot)
# u0, u1 = np.meshgrid(u0, u1)
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_trisurf(u0.flatten(), u1.flatten(), ctg_plot.flatten(), cmap=cm.jet, linewidth=0.05)
# # ax.tricontourf(u0.flatten(), u1.flatten(), ctg_plot.flatten(), zdir='z', cmap=cm.jet)
# ax.view_init(30, 45)
# ax.set_xlim(173, 373)
# ax.set_ylim(173, 373)
# ax.set_zlim(2500, 5500)
# ax.set_xlabel("$u_0$")
# ax.set_ylabel("$u_1$")
# ax.set_zlabel("Cost to go")
# ax.set_title("Controller actions against cost to go at initial uniform state of 273 K")
# plt.show()
# plt.savefig("Controller actions against cost to go at initial uniform state of 273 K.svg")
#
# # ----- LOCAL MINIMA
#
# plt.plot(u0, ctg_plot)
# plt.xlabel("$u_0$")
# plt.ylabel("Cost to go")
# plt.title("Cost against $u_0$, at initial state of 273 K \n $u_1$ fixed at 237 K")
# plt.annotate("Presence of local minima", (270, 3700))
#
# # NON DIFFERENTIABLE BEHAVIOUR
#
# plt.savefig("Presence of local minima.svg")
# plt.show()
#
# plt.plot(u0, ctg_plot)
# plt.xlabel("$u_0$")
# plt.ylabel("Cost to go")
# plt.title("Cost against $u_0$, at initial state of 273 K \n $u_1$ fixed at 237 K")
# plt.annotate("Non-differentiable behaviour", (270.15, 3621.28))
# plt.savefig("Non-differentiable behaviour.svg")
# plt.show()
# u0 = np.linspace(start=270, stop=270.5, num=200)
# u1 = 273
# start_time = python_timer.time()
#
# ctg_plot = []
# cst_plot = []
# for u0_i in u0:
#     pred_ctg, pred_cst = heatEq_nn.predict_ctg_cst(x, [u0_i, u1])
#     ctg_plot.append(pred_ctg)
#     cst_plot.append(pred_cst)
#
# print(python_timer.time() - start_time)