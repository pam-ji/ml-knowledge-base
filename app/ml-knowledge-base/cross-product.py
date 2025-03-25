import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from mpl_toolkits.mplot3d import Axes3D

def main():
    """
    Einfache Visualisierung des Vektorkreuzprodukts in 3D
    """
    st.subheader("3D Kreuzprodukt Visualisierung")
    
    # Interaktive Steuerelemente in der Seitenleiste
    st.sidebar.header("Vektoren einstellen")
    
    # Vektor 1
    st.sidebar.subheader("Vektor 1")
    v1_x = st.sidebar.slider("X-Komponente", -5.0, 5.0, 1.0, 0.1, key="v1x")
    v1_y = st.sidebar.slider("Y-Komponente", -5.0, 5.0, 2.0, 0.1, key="v1y")
    v1_z = st.sidebar.slider("Z-Komponente", -5.0, 5.0, 0.0, 0.1, key="v1z")
    
    # Vektor 2
    st.sidebar.subheader("Vektor 2")
    v2_x = st.sidebar.slider("X-Komponente", -5.0, 5.0, 2.0, 0.1, key="v2x")
    v2_y = st.sidebar.slider("Y-Komponente", -5.0, 5.0, 1.0, 0.1, key="v2y")
    v2_z = st.sidebar.slider("Z-Komponente", -5.0, 5.0, 0.0, 0.1, key="v2z")
    
    # Vektoren erstellen
    v1 = np.array([v1_x, v1_y, v1_z])
    v2 = np.array([v2_x, v2_y, v2_z])
    
    # Kreuzprodukt berechnen
    cross_product = np.cross(v1, v2)
    
    # Anzeigen der Vektorwerte
    st.write("### Vektorwerte")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Vektor 1:** [{v1_x}, {v1_y}, {v1_z}]")
        st.write(f"**Vektor 2:** [{v2_x}, {v2_y}, {v2_z}]")
    
    with col2:
        st.write(f"**Kreuzprodukt:** [{cross_product[0]:.2f}, {cross_product[1]:.2f}, {cross_product[2]:.2f}]")
    
    # 3D Ansicht Kontrollen
    st.sidebar.subheader("3D Ansicht")
    elev = st.sidebar.slider("Elevation", -90, 90, 30, 5)
    azim = st.sidebar.slider("Azimut", -180, 180, 30, 5)
    
    # 3D Abbildung erstellen
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Blickwinkel einstellen
    ax.view_init(elev=elev, azim=azim)
    
    # Vektoren als Pfeile zeichnen
    ax.quiver(0, 0, 0, v1[0], v1[1], v1[2], color='r', label='Vektor 1')
    ax.quiver(0, 0, 0, v2[0], v2[1], v2[2], color='b', label='Vektor 2')
    ax.quiver(0, 0, 0, cross_product[0], cross_product[1], cross_product[2], 
             color='g', label='Kreuzprodukt')
    
    # Ursprung markieren
    ax.scatter(0, 0, 0, color='black', s=50)
    
    # Achsenbeschriftungen
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Kreuzprodukt Visualisierung')
    ax.legend()
    
    # Plot in Streamlit anzeigen
    st.pyplot(fig)
    
    # Mathematische Erklärung
    st.subheader("Formel des Kreuzprodukts")
    st.latex(r"\vec{A} \times \vec{B} = |A||B|\sin(\theta)\hat{n}")
    
    # Komponenten-Form
    st.latex(r"\vec{A} \times \vec{B} = \begin{pmatrix} A_y B_z - A_z B_y \\ A_z B_x - A_x B_z \\ A_x B_y - A_y B_x \end{pmatrix}")
    
    # Kurze Erklärung
    st.write("""
    Das Kreuzprodukt zweier Vektoren:
    - Erzeugt einen Vektor, der senkrecht zu beiden Eingabevektoren steht
    - Die Länge entspricht der Fläche des Parallelogramms, das von beiden Vektoren aufgespannt wird
    - Die Richtung folgt der Rechte-Hand-Regel
    """)
    
    # Code-Snippet für das zweite Tab
    st.subheader("Python-Code Beispiel")
    st.code("""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Vektoren definieren
v1 = np.array([1, 2, 0])
v2 = np.array([2, 1, 0])

# Kreuzprodukt berechnen
cross = np.cross(v1, v2)
print(f"Kreuzprodukt: {cross}")

# 3D Plot erstellen
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Vektoren zeichnen
ax.quiver(0, 0, 0, v1[0], v1[1], v1[2], color='r', label='v1')
ax.quiver(0, 0, 0, v2[0], v2[1], v2[2], color='b', label='v2')
ax.quiver(0, 0, 0, cross[0], cross[1], cross[2], color='g', label='v1 × v2')

ax.set_xlim([-3, 3])
ax.set_ylim([-3, 3])
ax.set_zlim([-3, 3])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

plt.show()
""")

# Direct execution (for testing)
if __name__ == "__main__":
    main()