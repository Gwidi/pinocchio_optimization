import casadi as ca 
import matplotlib.pyplot as plt
import numpy as np

def robot_motion_optimization():
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

if __name__ == '__main__':
    robot_motion_optimization()