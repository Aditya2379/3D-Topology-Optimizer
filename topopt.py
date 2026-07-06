import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as splinalg
import matplotlib.pyplot as plt
import os

def main():
    nelx = 120
    nely = 40
    volfrac = 0.4
    penal = 3.0
    rmin = 1.5

    E0 = 1
    Emin = 1e-9
    nu = 0.3

    # Element stiffness matrix formulation
    A11 = np.array([[12, 3, -6, -3], [3, 12, 3, 0], [-6, 3, 12, -3], [-3, 0, -3, 12]])
    A12 = np.array([[-6, -3, 0, 3], [-3, -6, -3, -6], [0, -3, -6, 3], [3, -6, 3, -6]])
    B11 = np.array([[-4, 3, -2, 9], [3, -4, -9, 4], [-2, -9, -4, -3], [9, 4, -3, -4]])
    B12 = np.array([[2, -3, 4, -9], [-3, 2, 9, -2], [4, 9, 2, 3], [-9, -2, 3, 2]])
    KE = 1.0/(1.0-nu**2)/24.0 * (np.block([[A11, A12], [A12.T, A11]]) + nu*np.block([[B11, B12], [B12.T, B11]]))
    
    # Node numbering and DOFs
    edofMat = np.zeros((nelx*nely, 8), dtype=int)
    for elx in range(nelx):
        for ely in range(nely):
            el = ely + elx*nely
            n1 = (nely+1)*elx + ely
            n2 = (nely+1)*(elx+1) + ely
            edofMat[el, :] = np.array([2*n1, 2*n1+1, 2*n2, 2*n2+1, 2*n2+2, 2*n2+3, 2*n1+2, 2*n1+3])

    iK = np.kron(edofMat, np.ones((8,1))).flatten()
    jK = np.kron(edofMat, np.ones((1,8))).flatten()

    # Density Filter
    iH = np.ones(nelx*nely*(2*(int(np.ceil(rmin))-1)+1)**2)
    jH = np.ones(iH.shape)
    sH = np.zeros(iH.shape)
    k = 0
    for i1 in range(nelx):
        for j1 in range(nely):
            e1 = i1*nely + j1
            for i2 in range(max(i1-(int(np.ceil(rmin))-1), 0), min(i1+(int(np.ceil(rmin))), nelx)):
                for j2 in range(max(j1-(int(np.ceil(rmin))-1), 0), min(j1+(int(np.ceil(rmin))), nely)):
                    e2 = i2*nely + j2
                    iH[k] = e1
                    jH[k] = e2
                    sH[k] = max(0, rmin - np.sqrt((i1-i2)**2 + (j1-j2)**2))
                    k += 1
                    
    H = sps.coo_matrix((sH[:k], (iH[:k], jH[:k])), shape=(nelx*nely, nelx*nely)).tocsc()
    Hs = H.sum(1)

    # Boundary Conditions (MBB Beam half symmetric)
    F = np.zeros((2*(nely+1)*(nelx+1), 1))
    F[1, 0] = -1
    fixeddofs = np.union1d(np.arange(0, 2*(nely+1), 2), [2*(nelx+1)*(nely+1)-1])
    alldofs = np.arange(2*(nely+1)*(nelx+1))
    freedofs = np.setdiff1d(alldofs, fixeddofs)

    # Initialize density
    x = volfrac * np.ones(nely*nelx, dtype=float)
    change = 1.0
    loop = 0

    print("Starting Optimization...")
    while change > 0.01 and loop < 100:
        loop += 1
        
        # Assemble Global Stiffness
        sK = ((KE.flatten()[np.newaxis]).T * (Emin + x**penal * (E0 - Emin))).flatten(order='F')
        K = sps.coo_matrix((sK, (iK, jK)), shape=(2*(nely+1)*(nelx+1), 2*(nely+1)*(nelx+1))).tocsc()
        K = K[freedofs,:][:,freedofs]
        
        # FEA Solve
        U = np.zeros((2*(nely+1)*(nelx+1), 1))
        U[freedofs, 0] = splinalg.spsolve(K, F[freedofs])
        
        # Sensitivity
        U_edof = U[edofMat][:,:,0]
        ce = np.reshape(np.sum((U_edof @ KE) * U_edof, axis=1), (nely, nelx), order='F')
        c = np.sum((Emin + x**penal * (E0 - Emin)).reshape((nely, nelx), order='F') * ce)
        dc = -penal * x**(penal-1) * (E0 - Emin) * ce.flatten(order='F')
        
        # Apply Filter
        dc = np.asarray(H.dot(x * dc) / np.asarray(Hs).flatten() / np.maximum(0.001, x)).flatten()

        # Optimality Criteria (OC) update
        l1 = 0
        l2 = 1e9
        move = 0.2
        while (l2 - l1) > 1e-4:
            lmid = 0.5 * (l2 + l1)
            xnew = np.maximum(0.001, np.maximum(x - move, np.minimum(1, np.minimum(x + move, x * np.sqrt(-dc / lmid)))))
            if np.sum(xnew) - volfrac*nelx*nely > 0:
                l1 = lmid
            else:
                l2 = lmid
                
        change = np.max(np.abs(xnew - x))
        x = xnew
        print(f"Iter: {loop:3d} | Compliance: {c:8.4f} | Vol: {np.mean(x):.3f} | Change: {change:.4f}")

    print("Convergence reached. Saving image...")
    plt.figure(figsize=(10, 4))
    plt.imshow(-x.reshape((nely, nelx), order='F'), cmap='gray')
    plt.axis('off')
    
    # Save to artifacts directory so user can see it!
    artifact_path = r"C:\Users\adity\.gemini\antigravity\brain\588a7116-ac80-4b53-a692-8539ccd26852\topology_result.png"
    plt.savefig(artifact_path, bbox_inches='tight', pad_inches=0, dpi=300)
    print(f"Saved to {artifact_path}")

if __name__ == "__main__":
    main()
