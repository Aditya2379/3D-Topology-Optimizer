import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as splinalg
import trimesh
from skimage import measure
import scipy.ndimage as ndimage

def lk_3d(E, nu):
    pts = [-1/np.sqrt(3), 1/np.sqrt(3)]
    C = E / ((1 + nu) * (1 - 2 * nu)) * np.array([
        [1-nu, nu, nu, 0, 0, 0], [nu, 1-nu, nu, 0, 0, 0], [nu, nu, 1-nu, 0, 0, 0],
        [0, 0, 0, (1-2*nu)/2, 0, 0], [0, 0, 0, 0, (1-2*nu)/2, 0], [0, 0, 0, 0, 0, (1-2*nu)/2]
    ])
    KE = np.zeros((24, 24))
    for xi in pts:
        for eta in pts:
            for zeta in pts:
                dN = np.array([
                    [-(1-eta)*(1-zeta), (1-eta)*(1-zeta), (1+eta)*(1-zeta), -(1+eta)*(1-zeta), -(1-eta)*(1+zeta), (1-eta)*(1+zeta), (1+eta)*(1+zeta), -(1+eta)*(1+zeta)],
                    [-(1-xi)*(1-zeta), -(1+xi)*(1-zeta), (1+xi)*(1-zeta), (1-xi)*(1-zeta), -(1-xi)*(1+zeta), -(1+xi)*(1+zeta), (1+xi)*(1+zeta), (1-xi)*(1+zeta)],
                    [-(1-xi)*(1-eta), -(1+xi)*(1-eta), -(1+xi)*(1+eta), -(1-xi)*(1+eta), (1-xi)*(1-eta), (1+xi)*(1-eta), (1+xi)*(1+eta), (1-xi)*(1+eta)]
                ]) / 8.0
                B = np.zeros((6, 24))
                for i in range(8):
                    B[0, 3*i], B[1, 3*i+1], B[2, 3*i+2] = dN[0, i], dN[1, i], dN[2, i]
                    B[3, 3*i], B[3, 3*i+1] = dN[1, i], dN[0, i]
                    B[4, 3*i+1], B[4, 3*i+2] = dN[2, i], dN[1, i]
                    B[5, 3*i], B[5, 3*i+2] = dN[2, i], dN[0, i]
                KE += B.T @ C @ B
    return KE

def run():
    mesh = trimesh.load('input_beam.stl')
    bnd = mesh.extents
    scl = 30.0 / max(bnd)
    mesh.apply_scale(scl)
    mesh.apply_translation(-mesh.bounds[0])
    
    vox = mesh.voxelized(pitch=1.0)
    mat = vox.matrix
    nz, nx, ny = mat.shape
    if nx < 2 or ny < 2 or nz < 2:
        nz, nx, ny = 10, 30, 10
        mat = np.ones((nz, nx, ny), dtype=bool)

    vf, pnl, E0, Emin, nu = 0.3, 3.0, 1.0, 1e-9, 0.3
    KE = lk_3d(E0, nu)
    nele = nx * ny * nz
    edof = np.zeros((nele, 24), dtype=int)
    
    for k in range(nz):
        for i in range(nx):
            for j in range(ny):
                el = k*(nx*ny) + i*ny + j
                n1 = k*(nx+1)*(ny+1) + i*(ny+1) + j
                n2 = k*(nx+1)*(ny+1) + (i+1)*(ny+1) + j
                n3 = (k+1)*(nx+1)*(ny+1) + i*(ny+1) + j
                n4 = (k+1)*(nx+1)*(ny+1) + (i+1)*(ny+1) + j
                edof[el, :] = np.array([
                    3*n1, 3*n1+1, 3*n1+2, 3*n2, 3*n2+1, 3*n2+2, 3*n2+3, 3*n2+4, 3*n2+5, 3*n1+3, 3*n1+4, 3*n1+5,
                    3*n3, 3*n3+1, 3*n3+2, 3*n4, 3*n4+1, 3*n4+2, 3*n4+3, 3*n4+4, 3*n4+5, 3*n3+3, 3*n3+4, 3*n3+5
                ])

    iK = np.kron(edof, np.ones((24,1))).flatten()
    jK = np.kron(edof, np.ones((1,24))).flatten()

    ndof = 3 * (nx+1) * (ny+1) * (nz+1)
    F = np.zeros((ndof, 1))
    pas_el = ~mat.flatten()
    
    sol_idx = np.argwhere(mat)
    if len(sol_idx) == 0: return

    min_x, max_x = sol_idx[:, 1].min(), sol_idx[:, 1].max()
    sol_xmax = sol_idx[sol_idx[:, 1] == max_x]
    lz, lx, ly = sol_xmax[len(sol_xmax)//2]
    
    ld_el = lz*(nx*ny) + lx*ny + ly
    ld_nds = edof[ld_el, :]
    F[ld_nds[2], 0] = -1.0 
    
    fx_nds = []
    for k in range(nz+1):
        for j in range(ny+1):
            fx_nds.append(k*(nx+1)*(ny+1) + min_x*(ny+1) + j)
            
    fx_dofs = []
    for n in fx_nds:
        fx_dofs.extend([3*n, 3*n+1, 3*n+2])
    fx_dofs = np.array(fx_dofs)
    
    alldofs = np.arange(ndof)
    freedofs = np.setdiff1d(alldofs, fx_dofs)

    act_vol = np.sum(mat)
    tgt_v = vf * act_vol

    x = vf * np.ones(nele, dtype=float)
    x[pas_el] = 0.001 
    
    chg, lp = 1.0, 0
    while chg > 0.01 and lp < 35:
        lp += 1
        sK = ((KE.flatten()[np.newaxis]).T * (Emin + x**pnl * (E0 - Emin))).flatten(order='F')
        K = sps.coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
        K = K[freedofs,:][:,freedofs]
        
        U = np.zeros((ndof, 1))
        U[freedofs, 0] = splinalg.spsolve(K, F[freedofs])
        
        U_ed = U[edof][:,:,0]
        ce = np.sum((U_ed @ KE) * U_ed, axis=1)
        c = np.sum((Emin + x**pnl * (E0 - Emin)) * ce)
        dc = -pnl * x**(pnl-1) * (E0 - Emin) * ce
        
        dc3 = (dc * x).reshape((nz, nx, ny))
        dc3_flt = ndimage.uniform_filter(dc3, size=2, mode='constant')
        dc = dc3_flt.flatten() / np.maximum(0.001, x)
        dc[pas_el] = 0.0
        
        l1, l2, mv = 0, 1e9, 0.2
        while (l2 - l1) > 1e-4:
            lmid = 0.5 * (l2 + l1)
            xn = np.maximum(0.001, np.maximum(x - mv, np.minimum(1, np.minimum(x + mv, x * np.sqrt(np.abs(dc) / lmid)))))
            xn[pas_el] = 0.001 
            if np.sum(xn) - tgt_v > 0: l1 = lmid
            else: l2 = lmid
                
        chg = np.max(np.abs(xn - x))
        x = xn
        print(f"Iter: {lp:3d} | Obj: {c:8.4f} | Vol: {np.sum(x)/act_vol:.3f} | dX: {chg:.4f}")

    vol = x.reshape((nz, nx, ny))
    vol_sm = ndimage.gaussian_filter(vol, sigma=0.5)
    verts, faces, norms, vals = measure.marching_cubes(vol_sm, level=0.4)
    
    out_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    out_path = r"C:\Users\adity\.gemini\antigravity\brain\588a7116-ac80-4b53-a692-8539ccd26852\optimized_beam.stl"
    out_mesh.export(out_path)

if __name__ == "__main__":
    run()
