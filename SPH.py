import taichi as ti
import numpy as np
import math
import matplotlib.pyplot as plt
# from taichi.type.annotations import ext_arr

# ---------------------------------------------------------------------------- #
#                              constant paramters                              #
# ---------------------------------------------------------------------------- #
# global constant
numPar = 1000
density0 = 1.0
kernelRadius = 1.0
mass = 1.0
boundX = 100.0  # whole region size
boundY = 100.0
waterBoundX = 0.2 * boundX
waterBoundY = 0.2 * boundY
waterPosX = 0.3 * boundX
waterPosY = 0.2 * boundY
restiCoeff = 0.9
fricCoeff = 0.2
EosCoeff = 50.0  # coefficient of equation of state, copied from splishsplash
EosExponent = 7.0  # Gamma of equation of state
viscosity_mu = 1e-1  # dynamic viscosity coefficient, the mu
dt = 1e-4  # time step size
cellSize = 4.0  # not real cell, just for hashing
numCellX = ti.ceil(boundX/cellSize)   # number of cells in x direction
numCellY = ti.ceil(boundY/cellSize)   # number of cells in y direction
numCell = (numCellX)*(numCellY)
# number of cells, additional layer is to prevent out of bound

# ---------------------------------------------------------------------------- #
#                                end parameters                                #
# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
#                             BEGIN TAICHI PROGRAM                             #
# ---------------------------------------------------------------------------- #
DEBUG = False  # print debug info

if DEBUG == True:
    ti.init(arch=ti.cpu, debug=True, excepthook=True,
            cpu_max_num_threads=1, advanced_optimization=False)
else:
    ti.init(arch=ti.gpu)


paused = ti.field(dtype=ti.i32, shape=())
count = ti.field(int, shape=())


# ---------------------------------------------------------------------------- #
#                                physical fields                               #
# ---------------------------------------------------------------------------- #
# global field
position = ti.Vector.field(2, float, shape=numPar)
velocity = ti.Vector.field(2, float, shape=numPar)
density = ti.field(float, shape=numPar)
pressure = ti.field(float, shape=numPar)
acceleration = ti.Vector.field(2, float, shape=numPar)
pressureGradientForce = ti.Vector.field(2, float, shape=numPar)
viscosityForce = ti.Vector.field(2, float, shape=numPar)


# ---------------------------------------------------------------------------- #
#                            neighbor search variables                         #
# ---------------------------------------------------------------------------- #
# neighbor search related
maxNumNeighbors = 100  # max len for neiList and cell2Par
maxNumParInCell = 100

# par2Cell = ti.field(int, numPar)
# cell2Par = ti.field(int)
# ti.root.pointer(ti.i, numCell).dynamic(ti.j, maxListLen).place(cell2Par)
# neighborList = ti.field(int)
# ti.root.pointer(ti.i, numPar).dynamic(ti.j, maxListLen).place(neighborList)

numParInCell = ti.field(int, shape=numCell)
# how many particles in this cell
# usage: numParInCell[cellID]
neighbor = ti.field(int, shape=(numPar, maxNumNeighbors))
# the neighbor of the particle
# usage: neighbor[parID, kthPar], use with numNeighbor
numNeighbor = ti.field(int, shape=numPar)
# the length of neighborList of this particle
# usage: numNeighbor[parID]

cell2Par = ti.field(int, shape=(numCell, maxNumParInCell))
# convert a cell ID to particle IDs in this cell
# usage: cell2Par[cellID, kthParInThisCell], use with numCell2Par
numCell2Par = ti.field(int, shape=numPar)
# usage: cell2Par[cellID]
# ---------------------------------------------------------------------------- #
#                         end neighbor search variables                        #
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
#                          kernel Func and dirivatives                         #
# ---------------------------------------------------------------------------- #
@ti.func
def kernelFunc(r):
    # poly6 kernel, r is the distance(r>0)
    res = 0.0
    h = kernelRadius
    if r < kernelRadius:
        x = (h * h - r * r) / (h * h * h)
        res = 315.0 / 64.0 / math.pi * x * x * x
    return res


@ti.func
def firstDW(r):
    # first derivative of spiky kernel, r is the distance(r>0)
    res = 0.0
    h = kernelRadius
    if r < h:
        x = 1.0 - r / h
        res = -30.0 / (math.pi * h**3) * x * x
    return res


@ti.func
def secondDW(r):
    # second derivative of kernel W
    # r must be non-negative
    h = kernelRadius
    res = 0.0
    if r < kernelRadius:
        x = 1.0 - r / h
        res = 60.0 / (math.pi * h**4) * x
    return res
# ---------------------------------------------------------------------------- #
#                        end kernel Func and dirivatives                       #
# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
#                                neighbor search                               #
# ---------------------------------------------------------------------------- #


@ti.kernel
def neighborSearch():
    searchRadius = kernelRadius * 1.1
    # the search radius is not neccearily equals to the kernel radius
    # sometimes it> kernel radius, e.g. 1.1*kernelRadius

    # update hash table: cell2Par
    for par in range(numPar):
        cell = getCell(position[par])  # get cell ID from position

        k = ti.atomic_add(numParInCell[cell], 1)  # add the numParInCell by one
        # usage of ti.atomic_add： numParInCell[cell] add by one,
        #  old value of numParInCell[cell] is stored in k

        cell2Par[cell, k] = par

    # begin building the neighbor list
    for i in range(numPar):

        cell = getCell(position[i])
        # get cell ID from position

        kk = 0  # kk is the kkth neighbor in the neighbor list

        offs = ti.Vector([0, 1, -1,
                          -numCellX, -numCellX-1, -numCellX+1,
                          numCellX,  numCellX-1, numCellX+1])
        neiCellList = cell + offs
        # the 9 neighbor cells

        for ii in ti.static(range(9)):
            cellToCheck = neiCellList[ii]
            # which cell to check

            if IsInBound(cellToCheck):
                # prevent the cell ID out of bounds, which will happen in the boundary cells
                # after adding the offset. If out-of-bounds, do not check
                for k in range(numParInCell[cellToCheck]):
                    # kth particle in this cell

                    j = cell2Par[cellToCheck, k]
                    # j is another particle in this cell

                    if kk < maxNumNeighbors and j != i and \
                            (position[i] - position[j]).norm() < searchRadius:

                        # j is the kkth neighbor of particle i
                        neighbor[i, kk] = j

                        kk += 1
        numNeighbor[i] = kk


# the helper function for neighbor search
# check whether the cell ID out of bounds
@ti.func
def IsInBound(c):
    return c >= 0 and c < numCell


# the helper function for neighbor search
@ti.func
def getCell(pos):
    # 0.5 is for correct boundary cell ID, see doc for detail
    cellID = int(pos.x/cellSize - 0.5) + \
        int(pos.y/cellSize - 0.5)*numCellX

    # cellID=(position[i] / cellSize).cast(int)
    return cellID
# ---------------------------------------------------------------------------- #
#                              end neighbor search                             #
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
#                  main steps for computing physical variables                 #
# ---------------------------------------------------------------------------- #
@ti.kernel
def computeDensity():
    eps = 0.1  # to prevent the density be zero, because it is denominator
    for i in density:
        for k in range(numNeighbor[i]):
            j = neighbor[i, k]

            r = (position[i]-position[j]).norm()
            density[i] += mass * kernelFunc(r)

        # to prevent the density be zero, because it is denominator
        if density[i] < eps:
            density[i] = eps


@ti.kernel
def computeViscosityForce():
    for i in viscosityForce:
        for k in range(numNeighbor[i]):
            j = neighbor[i, k]

            r = (position[j]-position[i]).norm()

            viscosityForce[i] += mass * mass * viscosity_mu * (
                (velocity[j] - velocity[i]) / density[j]
            ) * secondDW(r)


@ti.kernel
def computePressure():
    den = 0.0
    for i in pressure:
        # prevent the negative scaling, if density<density0, then pressure=0
        den = ti.max(density[i], density0)
        pressure[i] = EosCoeff*((den / density0) ** EosExponent - 1.0)


@ti.kernel
def computePressureGradientForce():
    for i in pressureGradientForce:
        for k in range(numNeighbor[i]):
            j = neighbor[i, k]

            # grad Wij
            dir = (position[i]-position[j]).normalized()
            r = (position[i]-position[j]).norm()
            gradW = firstDW(r) * dir

            pressureGradientForce[i] -= mass * mass * (
                pressure[i] / density[i] ** 2 + pressure[j] / density[j] ** 2
            ) * gradW


@ti.kernel
def computeAcceleration():
    for i in acceleration:
        gravity = ti.Vector([0, -100])

        acceleration[i] += gravity  \
                        + viscosityForce[i] / mass  \
                        + pressureGradientForce[i] / mass


@ti.kernel
def advanceTime():
    for i in range(numPar):
        velocity[i] += acceleration[i] * dt
        position[i] += velocity[i] * dt


@ti.kernel
def boundaryCollision():
    eps = 0.1
    for i in range(numPar):
        # left
        if position[i].x < 0.0:
            position[i].x = 0.0
            velocity[i].x *= -restiCoeff
            # do not handle friction for debugging

        # right
        elif position[i].x >= boundX - eps:
            position[i].x = boundX - eps
            velocity[i].x *= -restiCoeff

        # top
        elif position[i].y >= boundY - eps:
            position[i].y = boundY - eps
            velocity[i].y *= -restiCoeff

        # bottom
        elif position[i].y < 0.0:
            position[i].y = 0.0
            velocity[i].y *= -restiCoeff


@ti.kernel
def pseudoViscosity():
    pass#TODO: not implemented yet



def clear():
    # clear the density
    density.fill(0.0)

    # clear the forces and acceleration
    acceleration.fill(0.0)
    pressureGradientForce.fill(0.0)
    viscosityForce.fill(0.0)

    # clear the neighbor list and cell2Par
    numParInCell.fill(0)
    numNeighbor.fill(0)
    neighbor.fill(-1)  # because the cell ID begin with 0, default should be -1
    cell2Par.fill(0)


def step():
    count[None] += 1

    clear()
    neighborSearch()
    computeDensity()
    computeViscosityForce()
    computePressure()
    computePressureGradientForce()
    computeAcceleration()
    advanceTime()
    boundaryCollision()

    # TESTPrintParticleInfo(0)
    # TESTPrintNeighbor(0)
    # TESTPrintParticleInfo(1)
    # TESTPrintParticleInfo(32)
# ---------------------------------------------------------------------------- #
#                                end main steps                                #
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
#                                     init                                     #
# ---------------------------------------------------------------------------- #

@ti.kernel
def initialization():
    for i in range(numPar):
        # aligned init:
        r = waterBoundX/waterBoundY
        a = waterBoundX*waterBoundY/numPar
        dx = ti.sqrt(a*r)
        dy = ti.sqrt(a/r)

        perRow = (waterBoundX/dx)

        position[i] = [
            i % perRow * dx + waterPosX,
            i // perRow * dy + waterPosY,
        ]


@ti.kernel
def initialization_random():
    for i in range(numPar):
        # random init
        position[i] = [
            ti.random()*waterBoundX+waterPosX,
            ti.random()*waterBoundY+waterPosY,
        ]

# ---------------------------------------------------------------------------- #
#                                     draw                                     #
# ---------------------------------------------------------------------------- #


def draw(gui):

    # normalize position data (in (0.0,1.0)) for drawing
    pos = position.to_numpy()
    pos[:, 0] *= 1.0 / boundX
    pos[:, 1] *= 1.0 / boundY

    # draw the particles
    gui.circles(pos,
                radius=3.0,
                )

    # highlight particle 0 with red
    gui.circle(pos[0],
               0xDC143C,
               radius=3.0,)

    gui.text(
        content=f'press space to pause',
        pos=(0, 0.99),
        color=0x0)

    gui.text(content="position[0]={}".format(position[0]),
             pos=(0, 0.95),
             color=0x0)
             
    # draw the mesh TODO: not implemented yet
    # for i in range(numCellX):
    #     for j in range(numCellY):
    #         gui.line(begin=[0.2, 0], end=[0.2, 1],
    #                  color=0xdc143c, radius=1.0)



# ---------------------------------------------------------------------------- #
#                                   test func                                  #
# ---------------------------------------------------------------------------- #
def TESTKernel():
    h = kernelRadius

    r = np.zeros(1000)
    y = np.zeros(1000)
    y1 = np.zeros(1000)
    y2 = np.zeros(1000)
    for i in range(1000):
        r[i] = i * h/1000
        y[i] = kernelFunc(r[i])
        y1[i] = firstDW(r[i])
        y2[i] = secondDW(r[i])
    plt.plot(r, y, 'r')
    plt.plot(r, y1, 'g')
    plt.plot(r, y2, 'b')
    plt.show()
    # np.savetxt("y.csv", y, delimiter=',')
    # np.savetxt("y1.csv", y1, delimiter=',')
    # np.savetxt("y2.csv", y2, delimiter=',')
    np.savetxt("kernelFunc.csv", [y, y1, y2], delimiter=',')


def TESTTwoPar():
    # init with only two particles
    # set numPar=2
    # cancel gravity
    gui = ti.GUI("SPHDamBreak",
                 background_color=0x112F41,
                 res=(1000, 1000)
                 )

    distX = 5*kernelRadius/10.0
    distY = 0.0  # 5*kernelRadius/10.0

    position[0] = [
        boundX / 2.0,
        boundY / 2.0,
    ]
    position[1] = [
        boundX/2.0 + distX,
        boundY/2.0 + distY,
    ]

    while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
        for s in range(1):
            step()
        draw(gui)


def TESTPrintNeighbor(i):
    print("########neighbor of ", i, '#########')
    print('numNeighbor=', numNeighbor[i])
    for k in range(numNeighbor[i]):
        j = neighbor[i, k]
        print('neighbor[{},{}]={}'.format(i, k, j))
        print('position[{}]={}'.format(j, position[j]))
        print('velocity[{}]={}'.format(j, velocity[j]))
    print("########END neighbor of ", i, '#########')


def TESTPrintParticleInfo(i):
    print('###########print particle info {}###########'.format(i))
    print('density[{}]={}'.format(i, density[i]))
    print('pressure[{}]={}'.format(i, pressure[i]))
    print('position[{}]={}'.format(i, position[i]))
    print('velocity[{}]={}'.format(i, velocity[i]))
    print('###########END print particle info {}###########'.format(i))


# ---------------------------------------------------------------------------- #
#                                 end test func                                #
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
#                                      run                                     #
# ---------------------------------------------------------------------------- #
def run():
    # constantly run
    gui = ti.GUI("SPHDamBreak",
                 background_color=0x112F41,
                 res=(1000, 1000)
                 )

    frame=0

    paused[None] = False

    while True:
        for e in gui.get_events(ti.GUI.PRESS):

            if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
                exit()

            # press space to pause
            if e.key == gui.SPACE:
                paused[None] = not paused[None]

            # press s to step once
            elif e.key == 's':
                for s in range(1):
                    step()

        if not paused[None]:
            for s in range(100):
                step()

        draw(gui)

        # gui.show(f'data/temp/{frame:06d}.png')
        # frame+=1
        gui.show()

 # ------------------------------- end func run ------------------------------- #


# ---------------------------------------------------------------------------- #
#                                     main                                     #
# ---------------------------------------------------------------------------- #

if __name__ == '__main__':

    # ----------------------------------- test ----------------------------------- #
    # TESTKernel()
    # TESTTwoPar()
    # TESTStepOnce(100)
    # np.savetxt("position.csv", position.to_numpy(), delimiter=',')

    # ------------------------------------ run ----------------------------------- #
    initialization()
    run()
