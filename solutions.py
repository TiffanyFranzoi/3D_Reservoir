import numpy as np


class Solucoes:
    def __init__(self, pi, pwf, mi, k, phi, ct, rw, req, c_pb, b, lx, ly, lz, rx, ry, rz, bx, by, bz, nx, my, kz,
                 delta_x, delta_y, delta_z, delta_t, bloco_poco_produt):
        """
        Inicializa a classe Solucoes com os parâmetros do problema.

        Args:
            pi (float): Pressão inicial [kgf/cm²].
            pwf (float): Pressão no poço [kgf/cm²].
            mi (float): Viscosidade do fluido [cP].
            k (float): Permeabilidade [mD].
            phi (float): Porosidade [-].
            ct (float): Compressibilidade total [(kgf/cm²)^-1].
            rw (float): Raio do poço [m].
            req (float): Raio equivalente do reservatório [m].
            c_pb (float): Constante Petrobras.
            b (float): Beta.
            lx, ly, lz (float): Comprimento do reservatório [m].
            rx, ry, rz (float): Variável auxiliar.
            bx, by, bz (float): Variável auxiliar.
            nx, my, kz (float): Quantidade de blocos em x, y e z.
            delta_x, delta_y, delta_z (float): Discretização do reservatório para x, y e z.
            delta_t (float): Discretização do tempo.
            bloco_poco_produt (float): Bloco da malha em que o poço está inserido.
        """
        # Properties
        self.pi = pi
        self.pwf = pwf
        self.mi = mi
        self.k = k
        self.phi = phi
        self.ct = ct
        self.rw = rw
        self.req = req
        self.c_pb = c_pb
        self.b = b
        self.Lx, self.Ly, self.Lz = lx, ly, lz
        self.rx, self.ry, self.rz = rx, ry, rz
        self.Bx, self.By, self.Bz = bx, by, bz
        # Simulation
        self.nx, self.my, self.kz = nx, my, kz
        self.delta_x, self.delta_y, self.delta_z = delta_x, delta_y, delta_z
        self.delta_t = delta_t
        self.bloco_poco_produt = bloco_poco_produt

    # ----------------- Equações dos blocos -----------------------
    def eq_blocos(self):
        """
        Calcula a matriz A de pressão em cada bloco.

        Returns:
            ndarray: Matriz de pressões calculadas.
        """
        A = np.zeros(shape=(self.nx * self.my * self.kz, self.nx * self.my * self.kz))

        for z in range(self.kz):

            # -------------------------- PRIMEIRA FACE -------------------------------------
            if z == 0:
                for i in range(self.nx * self.my):
                    # ----------- BLOCOS DE CANTOS ----------

                    # primeiro bloco (bloco 0) da face k = 0
                    if i == 0:
                        # permeabilidade equivalente
                        k_right = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i + 1]))) ** (-1)
                        k_down = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i + self. my]))) ** (-1)
                        k_costas = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i + (self.my * self.nx)]))) ** (-1)

                        # calculo dos coef das press
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx)] = (1 + self.By * k_down + self.Bx *
                                                                                       k_right + self.Bz * k_costas)
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) + 1] = -self.Bx * k_right
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) + self.my] = -self.By * k_down
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) + self.nx * self.my] = (-self.Bz *
                                                                                                           k_costas)

                    # bloco (n*m)-1   #ultimo bloco da face k = 0
                    if i == ((self.nx * self.my) - 1):
                        # permeabilidade equivalente
                        k_costas = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i + (self.my * self.nx)]))) ** (-1)
                        k_up = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i - self.my]))) ** (-1)
                        k_left = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i - 1]))) ** (-1)

                        # calculo dos coef das press
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) + self.nx * self.my] = (-self.Bz *
                                                                                                           k_costas)
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) - self.my] = -self.By * k_up
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) - 1] = -self.Bx * k_left
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx)] = (1 + self.By * k_up + self.Bx *
                                                                                       k_left + self.Bz * k_costas)

                    # bloco (m)-1 da face k = 0
                    if i == (self.my - 1):
                        # permeabilidade equivalente
                        k_left = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i - 1]))) ** (-1)
                        k_down = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i + self.my]))) ** (-1)
                        k_costas = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i + (self.my * self.nx)]))) ** (-1)

                        # calculo dos coef das press
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) - 1] = -self.Bx * k_left
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) + self.my] = -self.By * k_down
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) + self.nx * self.my] = (-self.Bz *
                                                                                                           k_costas)
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx)] = (1 + self.By * k_down + self.Bx *
                                                                                       k_left + self.Bz * k_costas)

                    # bloco (n*m)-(m) da face k = 0
                    if i == ((self.nx * self.my) - self.my):
                        # permeabilidade equivalente
                        k_right = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i + 1]))) ** (-1)
                        k_up = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i - self.my]))) ** (-1)
                        k_costas = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i + (self.my * self.nx)]))) ** (-1)

                        # calculo dos coef das press
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx)] = (1 + self.By * k_up + self.Bx *
                                                                                       k_right + self.Bz * k_costas)
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) + 1] = -self.Bx * k_right
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) - self.my] = -self.By * k_up
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) + self.nx * self.my] = (-self.Bz *
                                                                                                           k_costas)

                    # ----------- BLOCOS DO MEIO-CIMA ----------

                    if 0 < i < (self.my - 1):
                        # permeabilidade equivalente
                        k_left = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i - 1]))) ** (-1)
                        k_right = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i + 1]))) ** (-1)
                        k_down = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i + self.my]))) ** (-1)
                        k_costas = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i + (self.my * self.nx)]))) ** (-1)

                        # calculo dos coef das press
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) - 1] = -self.Bx * k_left
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx)] = (
                                1 + self.By * k_down + self.Bx * (k_left + k_right) + self.Bz * k_costas)
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) + 1] = -self.Bx * k_right
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) + self.my] = -self.By * k_down
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) + self.nx * self.my] = (-self.Bz *
                                                                                                           k_costas)

                    # ----------- BLOCOS DO MEIO-MEIO ----------

                    if self.my < i < (self.my * self.nx) - (self.my + 1):
                        # permeabilidade equivalente
                        k_up = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i - self.my]))) ** (-1)
                        k_left = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i - 1]))) ** (-1)
                        k_right = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i + 1]))) ** (-1)
                        k_down = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i + self.my]))) ** (-1)
                        k_costas = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i + (self.my * self.nx)]))) ** (-1)

                        # calculo dos coef das press
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) - self.my] = -self.By * k_up
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) - 1] = -self.Bx * k_left
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx)] = (
                                1 + self.By * (k_up + k_down) + self.Bx * (k_left + k_right) + self.Bz * k_costas)
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) + 1] = -self.Bx * k_right
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) + self.my] = -self.By * k_down
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) + self.nx * self.my] = (-self.Bz *
                                                                                                           k_costas)

                        # left middle e um multiplo da coluna
                        if i != 0 and i != ((self.nx * self.my) - self.my) and i % self.my == 0:
                            # i != 0 and i != down left and i divisivel por m
                            A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) - self.my] = 0
                            A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) - 1] = 0
                            A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx)] = 0
                            A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) + 1] = 0
                            A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) + self.my] = 0
                            A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) + self.nx * self.my] = 0

                        # right middle
                        if i != (self.my - 1) and i != ((self.nx * self.my) - 1) and (i + 1) % self.my == 0:
                            # i != upper right and i != down rright and i divisivel por
                            A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) - self.my] = 0
                            A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) - 1] = 0
                            A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx)] = 0
                            A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) + 1] = 0
                            A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) + self.my] = 0
                            A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) + self.nx * self.my] = 0

                    # ----------- BLOCOS DO MEIO-BAIXO ----------

                    if ((self.nx * self.my) - self.my) < i < ((self.nx * self.my) - 1):
                        # permeabilidade equivalente
                        k_up = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i - self.my]))) ** (-1)
                        k_left = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i - 1]))) ** (-1)
                        k_right = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i + 1]))) ** (-1)
                        k_costas = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i + (self.my * self.nx)]))) ** (-1)

                        # calculo dos coef das press
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) - self.my] = -self.By * k_up
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) - 1] = -self.Bx * k_left
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx)] = (
                                    1 + self.By * k_up + self.Bx * (k_left + k_right) + self.Bz * k_costas)
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) + 1] = -self.Bx * k_right
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) + self.nx * self.my] = (-self.Bz *
                                                                                                           k_costas)

                    # ----------- BLOCOS DO MEIO-ESQUERDA ----------

                    if i != 0 and i != ((self.nx * self.my) - self.my) and i % self.my == 0:
                        # i != 0 and i != down left and i divisivel por m
                        # permeabilidade equivalente
                        k_up = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i - self.my]))) ** (-1)
                        k_right = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i + 1]))) ** (-1)
                        k_down = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i + self.my]))) ** (-1)
                        k_costas = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i + (self.my * self.nx)]))) ** (-1)

                        # calculo dos coef das press
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) - self.my] = -self.By * k_up
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx)] = (
                                    1 + self.By * (k_up + k_down) + self.Bx * k_right + self.Bz * k_costas)
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) + 1] = -self.Bx * k_right
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) + self.my] = -self.By * k_down
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) + self.nx * self.my] = (-self.Bz *
                                                                                                           k_costas)

                    # ----------- BLOCOS DO MEIO-DIREITA ----------

                    if i != (self.my - 1) and i != ((self.nx * self.my) - 1) and (i + 1) % self.my == 0:
                        # i != upper right and i != down rright and i divisivel por

                        # permeabilidade equivalente
                        k_up = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i - self.my]))) ** (-1)
                        k_left = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i - 1]))) ** (-1)
                        k_down = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i + self.my]))) ** (-1)
                        k_costas = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i + (self.my * self.nx)]))) ** (-1)

                        # calculo dos coef das press
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) - self.my] = -self.By * k_up
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) - 1] = -self.Bx * k_left
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx)] = (
                                    1 + self.By * (k_up + k_down) + self.Bx * k_left + self.Bz * k_costas)
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) + self.my] = -self.By * k_down
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) + self.nx * self.my] = (-self.Bz *
                                                                                                           k_costas)

            # -------------------------- ÚLTIMA FACE -------------------------------------
            if z == self.kz - 1:
                for i in range(self.nx * self.my):
                    # ----------- BLOCOS DE CANTOS ----------

                    # primeiro bloco (bloco 0) da face k = kz
                    if i == 0:
                        # permeabilidade equivalente
                        k_right = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i + 1]))) ** (-1)
                        k_down = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i + self.my]))) ** (-1)
                        k_frente = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i - (self.my * self.nx)]))) ** (-1)

                        # calculo dos coef das press
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx)] = (1 + self.By * k_down + self.Bx *
                                                                                       k_right + self.Bz * k_frente)
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) + 1] = -self.Bx * k_right
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) + self.my] = -self.By * k_down
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) - self.nx * self.my] = (-self.Bz *
                                                                                                           k_frente)

                    # bloco (n*m)-1   #ultimo bloco da face k = kz
                    if i == ((self.nx * self.my) - 1):
                        # permeabilidade equivalente
                        k_frente = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i - (self.my * self.nx)]))) ** (-1)
                        k_up = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i - self.my]))) ** (-1)
                        k_left = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i - 1]))) ** (-1)

                        # calculo dos coef das press
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) - self.nx * self.my] = (-self.Bz *
                                                                                                           k_frente)
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) - self.my] = -self.By * k_up
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) - 1] = -self.Bx * k_left
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx)] = (1 + self.By * k_up + self.Bx *
                                                                                       k_left + self.Bz * k_frente)

                    # bloco (m)-1 da face k = kz
                    if i == (self.my - 1):
                        # permeabilidade equivalente
                        k_left = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i - 1]))) ** (-1)
                        k_down = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i + self.my]))) ** (-1)
                        k_frente = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i - (self.my * self.nx)]))) ** (-1)

                        # calculo dos coef das press
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) - 1] = -self.Bx * k_left
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) + self.my] = -self.By * k_down
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) - self.nx * self.my] = (-self.Bz *
                                                                                                           k_frente)
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx)] = (1 + self.By * k_down + self.Bx *
                                                                                       k_left + self.Bz * k_frente)

                    # bloco (n*m)-(m) da face k = kz
                    if i == ((self.nx * self.my) - self.my):
                        # permeabilidade equivalente
                        k_right = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i + 1]))) ** (-1)
                        k_up = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i - self.my]))) ** (-1)
                        k_frente = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i - (self.my * self.nx)]))) ** (-1)

                        # calculo dos coef das press
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx)] = (1 + self.By * k_up + self.Bx *
                                                                                       k_right + self.Bz * k_frente)
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) + 1] = -self.Bx * k_right
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) - self.my] = -self.By * k_up
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) - self.nx * self.my] = (-self.Bz *
                                                                                                           k_frente)

                    # ----------- BLOCOS DO MEIO ----------

                    if self.my < i < (self.my * self.nx) - (self.my + 1):
                        # permeabilidade equivalente
                        k_up = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i - self.my]))) ** (-1)
                        k_left = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i - 1]))) ** (-1)
                        k_right = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i + 1]))) ** (-1)
                        k_down = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i + self.my]))) ** (-1)
                        k_frente = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i - (self.my * self.nx)]))) ** (-1)

                        # calculo dos coef das press
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) - self.my] = -self.By * k_up
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) - 1] = -self.Bx * k_left
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx)] = (
                                1 + self.By * (k_up + k_down) + self.Bx * (k_left + k_right) + self.Bz * k_frente)
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) + 1] = -self.Bx * k_right
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) + self.my] = -self.By * k_down
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) - self.nx * self.my] = (-self.Bz *
                                                                                                           k_frente)

                        # left middle e um multiplo da coluna
                        if i != 0 and i != ((self.nx * self.my) - self.my) and i % self.my == 0:
                            # i != 0 and i != down left and i divisivel por m
                            A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) - self.nx * self.my] = 0
                            A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) - self.my] = 0
                            A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) - 1] = 0
                            A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx)] = 0
                            A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) + 1] = 0
                            A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) + self.my] = 0

                        # right middle
                        if i != (self.my - 1) and i != ((self.nx * self.my) - 1) and (i + 1) % self.my == 0:
                            # i != upper right and i != down rright and i divisivel por
                            A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) - self.nx * self.my] = 0
                            A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) - self.my] = 0
                            A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) - 1] = 0
                            A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx)] = 0
                            A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) + 1] = 0
                            A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) + self.my] = 0

                    # ----------- BLOCOS DO MEIO-CIMA ----------

                    if 0 < i < (self.my - 1):
                        # permeabilidade equivalente
                        k_left = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i - 1]))) ** (-1)
                        k_right = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i + 1]))) ** (-1)
                        k_down = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i + self.my]))) ** (-1)
                        k_frente = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i - (self.my * self.nx)]))) ** (-1)

                        # calculo dos coef das press
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) - self.nx * self.my] = (-self.Bz *
                                                                                                           k_frente)
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) - 1] = -self.Bx * k_left
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx)] = (
                                1 + self.By * k_down + self.Bx * (k_left + k_right) + self.Bz * k_frente)
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) + 1] = -self.Bx * k_right
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) + self.my] = -self.By * k_down

                    # ----------- BLOCOS DO MEIO-BAIXO ----------

                    if ((self.nx * self.my) - self.my) < i < ((self.nx * self.my) - 1):
                        # permeabilidade equivalente
                        k_frente = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i - (self.my * self.nx)]))) ** (-1)
                        k_up = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i - self.my]))) ** (-1)
                        k_left = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i - 1]))) ** (-1)
                        k_right = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i + 1]))) ** (-1)

                        # calculo dos coef das press
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) - self.nx * self.my] = (-self.Bz *
                                                                                                           k_frente)
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) - self.my] = -self.By * k_up
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) - 1] = -self.Bx * k_left
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx)] = (
                                    1 + self.By * k_up + self.Bx * (k_left + k_right) + self.Bz * k_frente)
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) + 1] = -self.Bx * k_right

                    # ----------- BLOCOS DO MEIO-ESQUERDA ----------

                    if i != 0 and i != ((self.nx * self.my) - self.my) and i % self.my == 0:
                        # i != 0 and i != down left and i divisivel por m
                        # permeabilidade equivalente
                        k_frente = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i - (self.my * self.nx)]))) ** (-1)
                        k_up = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i - self.my]))) ** (-1)
                        k_right = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i + 1]))) ** (-1)
                        k_down = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i + self.my]))) ** (-1)

                        # calculo dos coef das press
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) - self.nx * self.my] = (-self.Bz *
                                                                                                           k_frente)
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) - self.my] = -self.By * k_up
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx)] = (
                                    1 + self.By * (k_up + k_down) + self.Bx * k_right + self.Bz * k_frente)
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) + 1] = -self.Bx * k_right
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) + self.my] = -self.By * k_down

                    # ----------- BLOCOS DO MEIO-DIREITA ----------

                    if i != (self.my - 1) and i != ((self.nx * self.my) - 1) and (i + 1) % self.my == 0:
                        # i != upper right and i != down rright and i divisivel por

                        # permeabilidade equivalente
                        k_frente = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i - (self.my * self.nx)]))) ** (-1)
                        k_up = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i - self.my]))) ** (-1)
                        k_left = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i - 1]))) ** (-1)
                        k_down = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i + self.my]))) ** (-1)

                        # calculo dos coef das press
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) - self.nx * self.my] = (-self.Bz *
                                                                                                           k_frente)
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) - self.my] = -self.By * k_up
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) - 1] = -self.Bx * k_left
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx)] = (
                                    1 + self.By * (k_up + k_down) + self.Bx * k_left + self.Bz * k_frente)
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) + self.my] = -self.By * k_down

            # -------------------------- FACES INTERNAS -------------------------------------
            if 0 < z < self.kz - 1:
                for i in range(self.nx * self.my):
                    # ----------- BLOCOS DE CANTOS ----------

                    # primeiro bloco (bloco 0) da face k = meio
                    if i == 0:
                        # permeabilidade equivalente
                        k_right = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i + 1]))) ** (-1)
                        k_down = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i + self.my]))) ** (-1)
                        k_frente = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i - (self.my * self.nx)]))) ** (-1)
                        k_costas = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i + (self.my * self.nx)]))) ** (-1)

                        # calculo dos coef das press
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx)] = (
                                1 + self.By * k_down + self.Bx * k_right + self.Bz * (k_frente + k_costas))
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) + 1] = -self.Bx * k_right
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) + self.my] = -self.By * k_down
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) - self.nx * self.my] = (-self.Bz *
                                                                                                           k_frente)
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) + self.nx * self.my] = (-self.Bz *
                                                                                                           k_costas)

                    # bloco (n*m)-1   #ultimo bloco da face k = meio
                    if i == ((self.nx * self.my) - 1):
                        # permeabilidade equivalente
                        k_frente = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i - (self.my * self.nx)]))) ** (-1)
                        k_up = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i - self.my]))) ** (-1)
                        k_left = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i - 1]))) ** (-1)
                        k_costas = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i + (self.my * self.nx)]))) ** (-1)

                        # calculo dos coef das press
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) - self.nx * self.my] = (-self.Bz *
                                                                                                           k_frente)
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) + self.nx * self.my] = (-self.Bz *
                                                                                                           k_costas)
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) - self.my] = -self.By * k_up
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) - 1] = -self.Bx * k_left
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx)] = (
                                    1 + self.By * k_up + self.Bx * k_left + self.Bz * (k_frente + k_costas))

                    # bloco (m)-1 da face k = meio
                    if i == (self.my - 1):
                        # permeabilidade equivalente
                        k_left = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i - 1]))) ** (-1)
                        k_down = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i + self.my]))) ** (-1)
                        k_frente = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i - (self.my * self.nx)]))) ** (-1)
                        k_costas = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i + (self.my * self.nx)]))) ** (-1)

                        # calculo dos coef das press
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) - 1] = -self.Bx * k_left
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) + self.my] = -self.By * k_down
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) - self.nx * self.my] = (-self.Bz *
                                                                                                           k_frente)
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) + self.nx * self.my] = (-self.Bz *
                                                                                                           k_costas)
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx)] = (
                                1 + self.By * k_down + self.Bx * k_left + self.Bz * (k_frente + k_costas))

                    # bloco (n*m)-(m) da face k = meio
                    if i == ((self.nx * self.my) - self.my):
                        # permeabilidade equivalente
                        k_right = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i + 1]))) ** (-1)
                        k_up = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i - self.my]))) ** (-1)
                        k_frente = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i - (self.my * self.nx)]))) ** (-1)
                        k_costas = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i + (self.my * self.nx)]))) ** (-1)

                        # calculo dos coef das press
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx)] = (
                                1 + self.By * k_up + self.Bx * k_right + self.Bz * (k_frente + k_costas))
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) + 1] = -self.Bx * k_right
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) - self.my] = -self.By * k_up
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) - self.nx * self.my] = (-self.Bz *
                                                                                                           k_frente)
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) + self.nx * self.my] = (-self.Bz *
                                                                                                           k_costas)

                    # ----------- BLOCOS DE CIMA-MEIO ----------

                    if 0 < i < (self.my - 1):
                        # permeabilidade equivalente
                        k_left = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i - 1]))) ** (-1)
                        k_right = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i + 1]))) ** (-1)
                        k_down = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i + self.my]))) ** (-1)
                        k_frente = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i - (self.my * self.nx)]))) ** (-1)
                        k_costas = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i + (self.my * self.nx)]))) ** (-1)

                        # calculo dos coef das press
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) - self.nx * self.my] = (-self.Bz *
                                                                                                           k_frente)
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) - 1] = -self.Bx * k_left
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx)] = (
                                1 + self.By * k_down + self.Bx * (k_left + k_right) + self.Bz * (k_frente + k_costas))
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) + 1] = -self.Bx * k_right
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) + self.my] = -self.By * k_down
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) + self.nx * self.my] = (-self.Bz *
                                                                                                           k_costas)

                    # ----------- BLOCOS DO MEIO-BAIXO ----------

                    if ((self.nx * self.my) - self.my) < i < ((self.nx * self.my) - 1):
                        # permeabilidade equivalente
                        k_frente = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i - (self.my * self.nx)]))) ** (-1)
                        k_up = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i - self.my]))) ** (-1)
                        k_left = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i - 1]))) ** (-1)
                        k_right = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i + 1]))) ** (-1)
                        k_costas = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i + (self.my * self.nx)]))) ** (-1)

                        # calculo dos coef das press
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) - self.nx * self.my] = (-self.Bz *
                                                                                                           k_frente)
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) - self.my] = -self.By * k_up
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) - 1] = -self.Bx * k_left
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx)] = (
                                1 + self.By * k_up + self.Bx * (k_left + k_right) + self.Bz * (k_frente + k_costas))
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) + 1] = -self.Bx * k_right
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) + self.nx * self.my] = (-self.Bz *
                                                                                                           k_costas)
            # ----------- BLOCOS DO MEIO ----------

            for i in range(self.nx * self.my):

                # ----------- BLOCOS DO MEIO-MEIO ----------
                if self.my < i < (self.my * self.nx) - (self.my + 1) and 0 < z < (self.kz - 1):

                    # permeabilidade equivalente
                    k_frente = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i - (self.my * self.nx)]))) ** (-1)
                    k_costas = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i + (self.my * self.nx)]))) ** (-1)
                    k_up = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i - self.my]))) ** (-1)
                    k_down = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i + self.my]))) ** (-1)
                    k_right = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i + 1]))) ** (-1)
                    k_left = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i - 1]))) ** (-1)

                    # pressões
                    A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) - self.nx * self.my] = (-self.Bz *
                                                                                                       k_frente)
                    A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) - self.my] = -self.By * k_up
                    A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) - 1] = -self.Bx * k_left
                    A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx)] = (
                            1 + self.By * (k_up + k_down) + self.Bx * (k_left + k_right) + self.Bz * (k_frente +
                                                                                                      k_costas))
                    A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) + 1] = -self.Bx * k_right
                    A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) + self.my] = -self.By * k_down
                    A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) + self.nx * self.my] = (-self.Bz *
                                                                                                       k_costas)

                    # left middle e um multiplo da coluna
                    if i != 0 and i != ((self.nx * self.my) - self.my) and i % self.my == 0:
                        # i != 0 and i != down left and i divisivel por m
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) - self.nx * self.my] = 0
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) - self.my] = 0
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) - 1] = 0
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx)] = 0
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) + 1] = 0
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) + self.my] = 0
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) + self.nx * self.my] = 0

                    # right middle
                    if i != (self.my - 1) and i != ((self.nx * self.my) - 1) and (i + 1) % self.my == 0:
                        # i != upper right and i != down rright and i divisivel por
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) - self.nx * self.my] = 0
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) - self.my] = 0
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) - 1] = 0
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx)] = 0
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) + 1] = 0
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) + self.my] = 0
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) + self.nx * self.my] = 0

            # montando os blocos middle-left e middle-right das faces do meio
            if 0 < z < self.kz - 1:
                for i in range(self.nx * self.my):
                    # ----------- BLOCOS DO MEIO-ESQUERDA ----------

                    if i != 0 and i != ((self.nx * self.my) - self.my) and i % self.my == 0:
                        # i != 0 and i != down left and i divisivel por m
                        # permeabilidade equivalente
                        k_frente = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i - (self.my * self.nx)]))) ** (-1)
                        k_up = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i - self.my]))) ** (-1)
                        k_right = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i + 1]))) ** (-1)
                        k_down = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i + self.my]))) ** (-1)
                        k_costas = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i + (self.my * self.nx)]))) ** (-1)

                        # calculo dos coef das press
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) - self.nx * self.my] = (-self.Bz *
                                                                                                           k_frente)
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) - self.my] = -self.By * k_up
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx)] = (
                                1 + self.By * (k_up + k_down) + self.Bx * k_right + self.Bz * (k_frente + k_costas))
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) + 1] = -self.Bx * k_right
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) + self.my] = -self.By * k_down
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) + self.nx * self.my] = (-self.Bz *
                                                                                                           k_costas)

                    # ----------- BLOCOS DO MEIO-DIREITA ----------

                    if i != (self.my - 1) and i != ((self.nx * self.my) - 1) and (i + 1) % self.my == 0:
                        # i != upper right and i != down rright and i divisivel por

                        # permeabilidade equivalente
                        k_frente = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i - (self.my * self.nx)]))) ** (-1)
                        k_up = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i - self.my]))) ** (-1)
                        k_left = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i - 1]))) ** (-1)
                        k_down = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i + self.my]))) ** (-1)
                        k_costas = ((1 / 2) * ((1 / self.k[i]) + (1 / self.k[i + (self.my * self.nx)]))) ** (-1)

                        # calculo dos coef das press
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) - self.nx * self.my] = (-self.Bz *
                                                                                                           k_frente)
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) - self.my] = -self.By * k_up
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) - 1] = -self.Bx * k_left
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx)] = (
                                1 + self.By * (k_up + k_down) + self.Bx * k_left + self.Bz * (k_frente + k_costas))
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) + self.my] = -self.By * k_down
                        A[i + (z * self.my * self.nx)][i + (z * self.my * self.nx) + self.nx * self.my] = (-self.Bz *
                                                                                                           k_costas)

        return A

