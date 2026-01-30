import casadi as ca 
import matplotlib.pyplot as plt
import numpy as np
from robot import Robot
import pinocchio
from scipy.optimize import minimize 

def distance_minimization():
    '''Minimalizuj długość łamanej (20 odcinków) unikając kolizji z 3 kolistymi przeszkodami.'''
    opti = ca.Opti()

    n_points = 21 #20 ocinków = 21 punktów

    # Zmienne decyzyjne
    x = opti.variable(n_points)  # współrzędne x
    y = opti.variable(n_points)  # współrzędne y
    
    # Punkt początkowy i końcowy
    start = [0, 0]
    end = [20, 20]

    opti.subject_to(x[0] == start[0])
    opti.subject_to(y[0] == start[1])
    opti.subject_to(x[-1] == end[0])
    opti.subject_to(y[-1] == end[1])

    # Przeszkody: 3 okręgi o różnych promieniach
    obstacles = [
        {'center': [5, 5], 'radius': 2.5},    # Przeszkoda 1
        {'center': [10, 12], 'radius': 2.0},  # Przeszkoda 2
        {'center': [15, 8], 'radius': 1.5},   # Przeszkoda 3
    ]

     # Funkcja celu: minimalizuj całkowitą długość łamanej
    total_length = 0
    max_segment_length = 2.0
    for i in range(n_points - 1):
        dx = x[i+1] - x[i]
        dy = y[i+1] - y[i]
        segment_length = dx**2 + dy**2
        total_length += segment_length

        # Ograniczenie długości odcinka
        opti.subject_to(dx**2 + dy**2 <= max_segment_length**2)


        # Ograniczenie: punkty muszą znajdować się w odległości większej lub równej od promienia przeszkody + margines baezpieczeństwa
        for obs in obstacles:
            cx, cy = obs['center']
            r = obs['radius']
            r_sq = r**2
            r_padded_sq = r_sq + (segment_length / 2 )**2
            dist_sq = (x[i] - cx)**2 + (y[i] - cy)**2

            opti.subject_to(dist_sq >= r_padded_sq)
            dist_sq_next = (x[i+1] - cx)**2 + (y[i+1] - cy)**2
            
            
            opti.subject_to(dist_sq_next >= r_padded_sq)
    
    opti.minimize(total_length)


    initial_x = np.linspace(start[0], end[0], n_points)
    initial_y = np.linspace(start[1], end[1], n_points)
    
    # Dodajemy lekkie "wygięcie" startowe, żeby solver wiedział, w którą stronę ominąć pierwszą przeszkodę
    opti.set_initial(x, initial_x)
    opti.set_initial(y, initial_y + np.sin(np.linspace(0, np.pi, n_points)) * 2)
    
    # Rozwiąż
    opts = {
    'ipopt.print_level': 5,
    'ipopt.max_iter': 3000,
    'ipopt.tol': 1e-6,
    'ipopt.acceptable_tol': 1e-4,
    'ipopt.acceptable_iter': 15,
    'ipopt.mu_strategy': 'adaptive'
}
    opti.solver('ipopt', opts)
    sol = opti.solve()

    # Wyniki
    x_sol = [sol.value(x[i]) for i in range(n_points)]
    y_sol = [sol.value(y[i]) for i in range(n_points)]
    real_length = 0
    for i in range(len(x_sol) - 1):
        real_length += np.sqrt((x_sol[i+1] - x_sol[i])**2 + (y_sol[i+1] - y_sol[i])**2)


    plt.figure(figsize=(10, 10))

    # Narysuj przeszkody
    for obs in obstacles:
        circle = plt.Circle(obs['center'], obs['radius'], color='red', alpha=0.3, label='Obstacle')
        plt.gca().add_patch(circle)
    
    # Narysuj trajektorię
    plt.plot(x_sol, y_sol, 'b-o', linewidth=2, markersize=6, label='Path')
    plt.plot(start[0], start[1], 'go', markersize=12, label='Start')
    plt.plot(end[0], end[1], 'r*', markersize=15, label='Goal')
    
    plt.grid(True)
    plt.axis('equal')
    plt.xlim(-2, 22)
    plt.ylim(-2, 22)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title(f'Robot Path (Length: {real_length:.2f})')
    plt.show()
    
    print(f"Optimal path length: {real_length:.4f}")
    print(f"Number of segments: {n_points - 1}")



def robot_motion_optimization():
    """
    Optymalizacja ruchu robota: minimalizacja długości ścieżki końcówki roboczej
    z zachowaniem orientacji pionowej i osiągnięciem końca w zadanej kuli.
    """
    urdf_path = "iiwa_cup.urdf"
    robot = Robot(urdf_path=urdf_path)
    model = robot.model

    n_points = 21  # liczba punktów trajektorii (w tym start i koniec)
    nq = model.nq  # liczba stopni swobody

    opti = ca.Opti()
    Q = opti.variable(n_points, nq)  # zmienne decyzyjne: konfiguracje robota

    # Parametry zadania
    q_start = pinocchio.neutral(model)
    # Przykładowa kula celu (środek i promień w przestrzeni zadania)
    goal_center = np.array([0.6, 0.0, 0.7])
    goal_radius = 0.08

    # Ograniczenie: konfiguracja początkowa
    opti.subject_to([Q[0, j] == q_start[j] for j in range(nq)])

    # Ograniczenie: końcówka robocza w kuli na końcu
    for i in range(n_points):
        # Pozycja i orientacja końcówki roboczej dla każdego punktu
        q_i = Q[i, :]
        # Użyj CasADi funkcji do forward kinematics
        q_i_np = ca.MX.sym('q', nq)
        pinocchio.forwardKinematics(model, robot.data, q_i_np)
        pinocchio.updateFramePlacements(model, robot.data)
        ee_frame_id = model.getFrameId("F_link_ee")
        ee_pos = robot.data.oMf[ee_frame_id].translation

        # Zachowanie orientacji pionowej (oś Z narzędzia równoległa do Z świata)
        ee_rot = robot.data.oMf[ee_frame_id].rotation
        z_axis = ee_rot[:, 2]
        opti.subject_to(z_axis[0] == 0)
        opti.subject_to(z_axis[1] == 0)
        opti.subject_to(z_axis[2] >= 0.99)  # tolerancja

        if i == n_points - 1:
            # Ograniczenie końcowe: pozycja końcówki w kuli
            opti.subject_to(ca.sumsqr(ee_pos - goal_center) <= goal_radius**2)

    # Funkcja celu: suma długości odcinków w przestrzeni zadania
    total_length = 0
    for i in range(n_points - 1):
        total_length += ca.norm_2(Q[i+1, :] - Q[i, :])
    opti.minimize(total_length)

    opti.minimize(total_length)

    # Inicjalizacja (np. liniowa interpolacja w przestrzeni konfiguracyjnej)
    for j in range(nq):
        opti.set_initial(Q[:, j], np.linspace(q_start[j], q_start[j], n_points))

    # Rozwiązanie
    opts = {
        'ipopt.print_level': 3,
        'ipopt.max_iter': 1000,
        'ipopt.tol': 1e-4,
    }
    opti.solver('ipopt', opts)
    sol = opti.solve()

    Q_sol = np.array(opti.value(Q))
    print("Optymalna trajektoria w przestrzeni konfiguracyjnej:")
    print(Q_sol)

    # (Opcjonalnie) wizualizacja trajektorii końcówki roboczej
    ee_traj = []
    for i in range(n_points):
        pinocchio.forwardKinematics(model, robot.data, Q_sol[i, :])
        pinocchio.updateFramePlacements(model, robot.data)
        ee_traj.append(robot.data.oMf[ee_frame_id].translation.copy())
    ee_traj = np.array(ee_traj)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(ee_traj[:, 0], ee_traj[:, 1], ee_traj[:, 2], 'b.-', label='trajektoria EE')
    ax.scatter(goal_center[0], goal_center[1], goal_center[2], c='r', label='środek kuli celu')
    ax.legend()
    plt.show()


def robot_motion_optimization_scipy():
    """
    Optymalizacja ruchu robota: minimalizacja długości ścieżki końcówki roboczej
    z zachowaniem orientacji pionowej i osiągnięciem końca w zadanej kuli,
    z użyciem Pinocchio + scipy.optimize.minimize (wszystko na numpy).
    """
    urdf_path = "iiwa_cup.urdf"
    robot = Robot(urdf_path=urdf_path)
    model = robot.model
    data = robot.data

    n_points = 21  # liczba punktów trajektorii (w tym start i koniec)
    nq = model.nq  # liczba stopni swobody

    q_start = pinocchio.neutral(model)
    goal_center = np.array([0.6, 0.0, 0.7])
    goal_radius = 0.08
    ee_frame_id = model.getFrameId("F_link_ee")

    # Inicjalizacja: powiel q_start
    Q0 = np.tile(q_start, (n_points, 1))
    Q0_flat = Q0.flatten()

    def objective(Q_flat):
        Q = Q_flat.reshape((n_points, nq))
        total_length = 0
        for i in range(n_points - 1):
            pinocchio.forwardKinematics(model, data, Q[i])
            pinocchio.updateFramePlacements(model, data)
            ee1 = data.oMf[ee_frame_id].translation.copy()
            pinocchio.forwardKinematics(model, data, Q[i+1])
            pinocchio.updateFramePlacements(model, data)
            ee2 = data.oMf[ee_frame_id].translation.copy()
            total_length += np.linalg.norm(ee2 - ee1)
        return total_length

    def constraint_start(Q_flat):
        Q = Q_flat.reshape((n_points, nq))
        return Q[0] - q_start  # == 0

    def constraint_goal(Q_flat):
        Q = Q_flat.reshape((n_points, nq))
        pinocchio.forwardKinematics(model, data, Q[-1])
        pinocchio.updateFramePlacements(model, data)
        ee = data.oMf[ee_frame_id].translation
        return goal_radius**2 - np.sum((ee - goal_center)**2)  # >= 0

    def constraint_vertical(Q_flat):
        Q = Q_flat.reshape((n_points, nq))
        min_z = []
        for i in range(n_points):
            pinocchio.forwardKinematics(model, data, Q[i])
            pinocchio.updateFramePlacements(model, data)
            ee_rot = data.oMf[ee_frame_id].rotation
            z_axis = ee_rot[:, 2]
            # z_axis[0] == 0, z_axis[1] == 0, z_axis[2] >= 0.99
            min_z.append(z_axis[2] - 0.99)
        return np.array(min_z)  # >= 0

    def constraint_step(Q_flat):
        Q = Q_flat.reshape((n_points, nq))
        max_step = 0.2  # możesz dobrać wartość
        steps = []
        for i in range(n_points - 1):
            steps.append(max_step - np.linalg.norm(Q[i+1] - Q[i]))
        return np.array(steps)  # >= 0

    constraints = [
    {'type': 'eq', 'fun': constraint_start},
    {'type': 'ineq', 'fun': constraint_goal},
    {'type': 'ineq', 'fun': constraint_vertical},
    {'type': 'ineq', 'fun': constraint_step},
    ]

    result = minimize(
        objective,
        Q0_flat,
        method='SLSQP',
        constraints=constraints,
        options={'maxiter': 200, 'ftol': 1e-3, 'disp': True}
    )

    Q_sol = result.x.reshape((n_points, nq))
    print("Optymalna trajektoria w przestrzeni konfiguracyjnej:")
    print(Q_sol)

    # Wizualizacja trajektorii końcówki roboczej
    ee_traj = []
    for i in range(n_points):
        pinocchio.forwardKinematics(model, data, Q_sol[i, :])
        pinocchio.updateFramePlacements(model, data)
        ee_traj.append(data.oMf[ee_frame_id].translation.copy())
    ee_traj = np.array(ee_traj)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(ee_traj[:, 0], ee_traj[:, 1], ee_traj[:, 2], 'b.-', label='trajektoria EE')
    ax.scatter(goal_center[0], goal_center[1], goal_center[2], c='r', label='środek kuli celu')
    ax.legend()
    plt.show()


if __name__ == '__main__':
    # distance_minimization()
    robot_motion_optimization_scipy()