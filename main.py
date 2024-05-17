import pandas as pd
import numpy as np
import scipy as sc
import pyvista as pv
from solutions import Solucoes

# -------------------------------- Read Excel ------------------------------------
url = r'./5x5.xlsx'
excel = pd.ExcelFile(url)
abas = excel.sheet_names
quantidade_abas = len(abas)
# print(f"O arquivo Excel tem {quantidade_abas} abas.")

nx, my, kz = quantidade_abas, quantidade_abas, quantidade_abas

# --------------------------------- Constantes -----------------------------------
bloco_poco_produt = 124
pi = 1300
pwf = -900
rw = 0.25
phi = 0.2
mi = 2.1
ct = 9 * 10 ** (-7)
lx, ly, lz = 800, 800, 800
delta_t = 0.0005
c_pb = 3.484 * 10 ** (-4)
b = c_pb / (phi * mi * ct)
delta_x, delta_y, delta_z = lx / nx, ly / my, lz / kz
ry = delta_t / (delta_y ** 2)
rx = delta_t / (delta_x ** 2)
rz = delta_t / (delta_z ** 2)
by = b * ry
bx = b * rx
bz = b * rz
req = (delta_x * delta_y / np.pi) ** (1 / 2)

# ----------------------- Matriz Permeabilidade ----------------------------------
k = []
for i in range(kz - 1, -1, -1):
    K = pd.read_excel(url, sheet_name=i)
    kk = np.reshape(K.values, nx * my)
    k = np.append(kk, k)

# ------------------------ Plot da Permeabilidade ---------------------------------
# Reshape k into a 3D array
k_3d = np.reshape(k, (nx, my, kz))

grid = pv.ImageData()
grid.dimensions = np.array(k_3d.shape) + 1
grid.origin = (0, 0, 0)
grid.spacing = (5, 5, 5)

# Adiciona data values a cell data
grid.cell_data["Permeability"] = k_3d.flatten(order="F")

plotter = pv.Plotter()
sargs = dict(height=0.7, width=0.1, vertical=True, position_x=0.8, position_y=0.2, fmt="%1.1e",
             title_font_size=20, color="black", label_font_size=15)

plotter.add_mesh(grid, show_edges=True, cmap='jet', scalar_bar_args=sargs, smooth_shading=True)
plotter.add_text('Permeability [D]', position='upper_left', font_size=12, color='black')
plotter.show()



# ----------------- Inicializa as variáveis da classe -----------------------------
var = Solucoes(pi, pwf, mi, k, phi, ct, rw, req, c_pb, b, lx, ly, lz, rx, ry, rz, bx, by, bz, nx, my, kz, delta_x,
               delta_y, delta_z, delta_t, bloco_poco_produt)

# --------------------- Matriz de pressão na esquerda do poço ----------------------
poco_esq = np.zeros(shape=(nx * my * kz, nx * my * kz))

den = np.log(req / rw) * delta_x * delta_y
num = ((k[bloco_poco_produt]) / (phi * mi * ct))
vazao_poco_ladoesquerdo = -delta_t * (2 * np.pi * num) / den

poco_esq[bloco_poco_produt][bloco_poco_produt] = vazao_poco_ladoesquerdo

# ------------------------ Matriz de pressões A ---------------------------------
A = var.eq_blocos()
A_press = np.zeros(shape=(nx * my * kz, nx * my * kz))
for i in range(nx * my * kz):
    for j in range(nx * my * kz):
        A_press[i][j] = A[i][j] + poco_esq[i][j]

# ------------------------ Matriz de pressão na direita do poço -------------------
poco_direita = np.zeros(shape=(nx * my * kz))

aa = 2 * np.pi * ((k[bloco_poco_produt]) / (phi * mi * ct))
bb = np.log(req / rw) * delta_x * delta_y
vazao_poco_ladodireito = delta_t * pwf * aa / bb
poco_direita[bloco_poco_produt] = vazao_poco_ladodireito

soma_cont = np.zeros(shape=(nx * my * kz))  # somando o contorno
for i in range(nx * my * kz):
    soma_cont[i] = poco_direita[i]

press_t = np.zeros(shape=(nx * my * kz))  # pressao no tempo t
for i in range(nx * my * kz):
    press_t[i] = pi

# -------------------------------- Vetor b ---------------------------------
h = delta_x
Bo = 1
t_final = 5
vet_tempo = np.arange(0, t_final, delta_t)
b = np.zeros(shape=(nx * my * kz))
times_to_plot = [0, 3500, 6000, 9500]  # Tempo pra criar os subplots
plots = []

for t in range(len(vet_tempo)):

    # montando o vetor b
    for i in range(nx * my * kz):
        b[i] = press_t[i] + soma_cont[i]

    # calculando a pressão
    press_t = sc.linalg.solve(A_press, b)
    print(t)  # esse print serve somente para indicar em que t está rodando

    # --------------------- Plot pra tempos específicos -------------------------------
    if t in times_to_plot:
        a = np.reshape(press_t, (nx, my, kz))

        grid = pv.ImageData()  # Referência espacial
        grid.dimensions = np.array(a.shape) + 1  # Dimensão do grid: shape+1 pq coloca o valor no cell data
        grid.origin = (0, 0, 0)  # Origem (bottom left corner)
        grid.spacing = (5, 5, 5)  # Tamanho das células

        # Adiciona data values a cell data
        grid.cell_data["Pressure [kgf/cm²]"] = a.flatten(order="F")  # Flatten the array

        plots.append(grid)

# --------------------------- Plot do 3D com Subplots ------------------------------------
plotter = pv.Plotter(shape=(2, 2))  # 2x2

sargs = dict(height=0.7, width=0.1, vertical=True, position_x=0.8, position_y=0.2, fmt="%1.1e",
             title_font_size=20, color="black", label_font_size=15)

for i, grid in enumerate(plots):
    plotter.subplot(i // 2, i % 2)  # Posição do subplot
    plotter.add_mesh(grid, show_edges=True, cmap='jet', scalar_bar_args=sargs, clim=[-pwf, pi], smooth_shading=True)
    plotter.add_text(f'Time: {times_to_plot[i]:.1f}', position='upper_left', font_size=12, color='black')

plotter.link_views()
plotter.show()
