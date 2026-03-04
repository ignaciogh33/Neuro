"""
Genera gráficas de las fronteras de decisión para los problemas lógicos
(AND, OR, NAND, XOR) a partir de los CSVs generados por entrenar_logicos.py.

Cada figura tiene 2 subplots: Perceptrón (izquierda) y Adaline (derecha).
Los puntos se colorean según la clase esperada y se superpone la recta
de frontera de decisión cuando la red converge.
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOGICOS_DIR = os.path.join(SCRIPT_DIR, 'logicos_files')

PROBLEMAS = ['and', 'or', 'nand', 'xor']
REDES = ['perceptron', 'adaline']

COLORES = {1: '#2196F3', -1: '#F44336'}       # azul = 1, rojo = -1
MARCADORES = {1: 'o', -1: 's'}                 # círculo = 1, cuadrado = -1


def leer_csv_frontera(filepath):
    """Lee un CSV de frontera y devuelve metadatos + puntos."""
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        cabecera_meta = next(reader)   # problema,red,w1,w2,sesgo,converge
        fila_meta = next(reader)
        next(reader)                   # línea vacía
        cabecera_puntos = next(reader) # x1,x2,esperado,obtenido

        nombre = fila_meta[0]
        red = fila_meta[1]
        converge = fila_meta[5].strip() == 'True'
        w1 = float(fila_meta[2]) if fila_meta[2] else None
        w2 = float(fila_meta[3]) if fila_meta[3] else None
        sesgo = float(fila_meta[4]) if fila_meta[4] else None

        puntos = []
        for row in reader:
            if not row or not row[0]:
                continue
            x1, x2 = float(row[0]), float(row[1])
            esperado = int(float(row[2]))
            obtenido = int(float(row[3])) if row[3] else None
            puntos.append((x1, x2, esperado, obtenido))

    return {
        'nombre': nombre,
        'red': red,
        'converge': converge,
        'w1': w1, 'w2': w2, 'sesgo': sesgo,
        'puntos': puntos
    }


def dibujar_subplot(ax, datos):
    """Dibuja un subplot con los puntos y la frontera de decisión."""
    titulo_red = 'Perceptrón' if datos['red'] == 'perceptron' else 'Adaline'
    ax.set_title(titulo_red, fontsize=13, fontweight='bold')
    ax.set_xlabel('x₁', fontsize=11)
    ax.set_ylabel('x₂', fontsize=11)

    # Dibujar puntos por clase esperada
    for x1, x2, esperado, _ in datos['puntos']:
        ax.scatter(x1, x2, c=COLORES[esperado], marker=MARCADORES[esperado],
                   s=120, edgecolors='black', linewidths=0.8, zorder=5)

    # Rango del gráfico con margen
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(-0.3, 1.3)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--')

    # Dibujar frontera si hay datos de pesos (aunque no converja)
    if datos['w1'] is not None:
        w1, w2, sesgo = datos['w1'], datos['w2'], datos['sesgo']
        x_range = np.linspace(-0.5, 1.5, 300)

        line_style = '-' if datos['converge'] else '--'
        line_color = '#4CAF50' if datos['converge'] else '#FF9800'

        if abs(w2) > 1e-10:
            y_line = -(w1 * x_range + sesgo) / w2
            ax.plot(x_range, y_line, color=line_color, linewidth=2.5,
                    linestyle=line_style, label='Frontera', zorder=3)
        else:
            x_vert = -sesgo / w1
            ax.axvline(x=x_vert, color=line_color, linewidth=2.5,
                       linestyle=line_style, label='Frontera', zorder=3)

        if abs(w2) > 1e-10:
            ax.fill_between(x_range, y_line, 1.5,
                            alpha=0.08, color='#2196F3', zorder=1)
            ax.fill_between(x_range, -0.5, y_line,
                            alpha=0.08, color='#F44336', zorder=1)

        ax.legend(fontsize=9, loc='upper right')

        eq = f'{w1:.2f}·x₁ + {w2:.2f}·x₂ + ({sesgo:.2f}) = 0'
        ax.text(0.5, -0.15, eq, transform=ax.transAxes,
                ha='center', fontsize=8, fontstyle='italic', color='gray')

    # Mostrar "No converge" si no convergió (con o sin frontera)
    if not datos['converge']:
        ax.text(0.5, 0.5, 'No converge', transform=ax.transAxes,
                ha='center', va='center', fontsize=14,
                fontweight='bold', color='red', alpha=0.5)


def main():
    # Leyenda personalizada (fuera de los subplots)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2196F3',
               markersize=10, markeredgecolor='black', label='Clase +1'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#F44336',
               markersize=10, markeredgecolor='black', label='Clase -1'),
        Line2D([0], [0], color='#4CAF50', linewidth=2.5, label='Frontera'),
    ]

    for problema in PROBLEMAS:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        fig.suptitle(f'Frontera de decisión — {problema.upper()}',
                     fontsize=15, fontweight='bold', y=1.02)

        for idx, red in enumerate(REDES):
            csv_path = os.path.join(LOGICOS_DIR, f'{problema}_{red}.csv')
            if not os.path.exists(csv_path):
                print(f'⚠ No se encontró: {csv_path}')
                axes[idx].text(0.5, 0.5, 'CSV no encontrado',
                               transform=axes[idx].transAxes,
                               ha='center', va='center', fontsize=12, color='red')
                continue

            datos = leer_csv_frontera(csv_path)
            dibujar_subplot(axes[idx], datos)

        fig.legend(handles=legend_elements, loc='lower center',
                   ncol=3, fontsize=10, frameon=True,
                   bbox_to_anchor=(0.5, -0.05))

        plt.tight_layout()
        output_path = os.path.join(LOGICOS_DIR, f'{problema}_fronteras.png')
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f'✓ Guardada: {output_path}')
        plt.close(fig)

    print('\n¡Todas las gráficas generadas en logicos_files/!')


if __name__ == '__main__':
    main()
