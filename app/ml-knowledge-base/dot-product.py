import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

def main():
    """
    Einfache Visualisierung des Skalarprodukts (Dot Product) von Vektoren
    """
    st.subheader("Skalarprodukt Visualisierung")
    
    # Interaktive Steuerelemente in der Seitenleiste
    st.sidebar.header("Vektoren einstellen")
    
    # Vektor 1
    st.sidebar.subheader("Vektor 1")
    v1_x = st.sidebar.slider("X-Komponente", -5.0, 5.0, 3.0, 0.1)
    v1_y = st.sidebar.slider("Y-Komponente", -5.0, 5.0, 2.0, 0.1)
    
    # Vektor 2
    st.sidebar.subheader("Vektor 2")
    v2_x = st.sidebar.slider("X-Komponente", -5.0, 5.0, 2.0, 0.1)
    v2_y = st.sidebar.slider("Y-Komponente", -5.0, 5.0, 1.0, 0.1)
    
    # Vektoren erstellen
    v1 = np.array([v1_x, v1_y])
    v2 = np.array([v2_x, v2_y])
    
    # Länge der Vektoren berechnen
    v1_mag = np.linalg.norm(v1)
    v2_mag = np.linalg.norm(v2)
    
    # Skalarprodukt berechnen
    dot_product = np.dot(v1, v2)
    
    # Winkel zwischen den Vektoren
    if v1_mag > 0 and v2_mag > 0:
        cos_angle = dot_product / (v1_mag * v2_mag)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Numerische Fehler vermeiden
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)
    else:
        angle_deg = 0
    
    # Vektor-Informationen anzeigen
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Vektorwerte")
        st.write(f"**Vektor 1:** [{v1_x}, {v1_y}]")
        st.write(f"**Vektor 2:** [{v2_x}, {v2_y}]")
    
    with col2:
        st.write("### Berechnungsergebnisse")
        st.write(f"**Skalarprodukt:** {dot_product:.2f}")
        st.write(f"**Winkel zwischen Vektoren:** {angle_deg:.2f}°")
        
        # Besondere Fälle kennzeichnen
        if abs(dot_product) < 0.01:
            st.success("Die Vektoren stehen annähernd senkrecht zueinander")
        elif abs(abs(cos_angle) - 1) < 0.01:
            st.success("Die Vektoren sind annähernd parallel")
    
    # Abbildung erstellen
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Vektoren zeichnen
    ax.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', 
             scale=1, color='r', label='Vektor 1')
    ax.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', 
             scale=1, color='b', label='Vektor 2')
    
    # Projektion des zweiten Vektors auf den ersten zeigen
    if v1_mag > 0:
        proj_scalar = dot_product / (v1_mag * v1_mag)
        proj_vector = proj_scalar * v1
        ax.quiver(0, 0, proj_vector[0], proj_vector[1], angles='xy', scale_units='xy', 
                 scale=1, color='g', linestyle='--', label='Projektion')
    
    # Ursprung markieren
    ax.scatter(0, 0, color='black', s=50)
    
    # Achsenbeschriftungen
    max_val = max(abs(v1_x), abs(v1_y), abs(v2_x), abs(v2_y), 1) + 0.5
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True)
    ax.legend()
    ax.set_title('Skalarprodukt Visualisierung')
    ax.set_aspect('equal')
    
    # Plot in Streamlit anzeigen
    st.pyplot(fig)
    
    # Mathematische Erklärung
    st.subheader("Formel des Skalarprodukts")
    st.latex(r"A \cdot B = |A| |B| \cos(\theta)")
    st.latex(r"A \cdot B = A_x \times B_x + A_y \times B_y")
    
    # Kurze Erklärung
    st.write("""
    Das Skalarprodukt hat mehrere Bedeutungen:
    - Es misst, wie parallel zwei Vektoren zueinander sind
    - Es entspricht dem Produkt der Vektorlängen und dem Kosinus des Winkels zwischen ihnen
    - Es ist die Summe der Produkte der entsprechenden Komponenten
    - Es stellt die Projektion eines Vektors auf einen anderen dar, multipliziert mit der Länge des zweiten Vektors
    """)
    
    # Code-Beispiel für das zweite Tab
    st.subheader("Python-Code Beispiel")
    st.code("""
import numpy as np
import matplotlib.pyplot as plt

# Vektoren definieren
v1 = np.array([3, 2])
v2 = np.array([2, 1])

# Skalarprodukt berechnen
dot_product = np.dot(v1, v2)
print(f"Skalarprodukt: {dot_product}")

# Länge der Vektoren berechnen
v1_mag = np.linalg.norm(v1)
v2_mag = np.linalg.norm(v2)

# Winkel zwischen den Vektoren berechnen
cos_angle = dot_product / (v1_mag * v2_mag)
angle_rad = np.arccos(cos_angle)
angle_deg = np.degrees(angle_rad)
print(f"Winkel: {angle_deg:.2f}°")

# Plot erstellen
plt.figure(figsize=(8, 6))
plt.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', 
          scale=1, color='r', label='v1')
plt.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', 
          scale=1, color='b', label='v2')
plt.grid(True)
plt.xlim(-1, 5)
plt.ylim(-1, 5)
plt.legend()
plt.axis('equal')
plt.show()
""")

# Direct execution (for testing)
if __name__ == "__main__":
    main()
