# Sistema para Detectar la Velocidad de Coches

Sistema que detecte la velocidad de los coches.

### Desafíos principales:

1. **Calibración de distancia**:

   - Convertir distancias en píxeles a metros reales
   - Solución: Usar las marcas viales del carril como referencia de medida

2. **Distorsión de perspectiva**:

   - Los coches parecen moverse a diferentes velocidades según su posición
   - Solución: Aplicar transformación de perspectiva (bird's eye view)

3. **Oclusiones parciales**:
   - Coches que se tapan entre sí
   - Solución: Predecir trayectorias durante oclusiones breves

### Algoritmo propuesto:

1. Detectar y segmentar coches en cada frame
2. Aplicar un algoritmo de seguimiento para mantener identidad de cada coche
3. Para cada coche seguido:
   - Calcular centroide en cada frame
   - Aplicar corrección de perspectiva
   - Calcular distancia recorrida entre frames
   - Calcular velocidad = distancia/tiempo
4. Aplicar suavizado temporal para reducir ruido en las mediciones
