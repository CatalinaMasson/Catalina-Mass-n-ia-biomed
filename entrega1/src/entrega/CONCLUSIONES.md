# Conclusiones - Práctica LLMs para Biomedicina

**Nombre:**
**Fecha:**

---

## Ejercicio 1: Primera Llamada

### 1. Diferencia entre respuesta sin y con system instruction
#### Con System Instruction, el modelo sigue ciertas características que le indicamos, como por ejemplo el rol que debe asumir, el lenguaje que debe usar, etc. Sin System Instruction, el modelo responde de manera más general, con la información que tiene en su entrenamiento, sin tener en cuenta ningún contexto adicional.

### 2. ¿Pudiste modificar los parámetros internos del modelo? ¿Qué sí controlaste?
#### No pudimos modificar los parámetros internos del modelo, ya que son parámetros que se aprenden durante el entrenamiento del modelo y no pueden ser modificados por el usuario. Lo que sí pudimos controlar fue el prompt, el system instruction, el modelo elegido, etc.

### 3. ¿Qué pasaría si cambiaras el rol en el system instruction?
#### Si cambiaramos el rol en el system instruction, el modelo respondería de manera diferente, ya que estaría adoptando un rol distinto. Cambiaría su manera de expresar sus conocimientos, el lenguaje que usaría, etc.

### 4. ¿Qué system instruction sería útil para tu campo de estudio?
#### A nuestro parecer, creemos que sería útil un System Instruction que indique nuestro rol de Ingenieras Biomédicas, resaltando el trabajo interdiciplinario con otros profesionales como podrían ser médicos clínicos, bioinformáticos, biólogos, etc. Además, sería útil que el System Instruction indique que el modelo debe responder de manera clara y concisa, evitando jerga técnica de ser innecesaria.

---

## Ejercicio 2: Hiperparámetros

### 1. ¿Qué temperature usarías para un informe médico? ¿Y para brainstorming?
#### Para un informe médico usaríamos una temperature baja, alrededor de 0.0, para que las respuestas sean más conservadoras y consistentes. Para brainstorming usaríamos una temperature alta, alrededor de 1.5, para que las respuestas sean más creativas y variables.

### 2. ¿Qué pasó con maxOutputTokens=50? ¿Fue útil?
#### Con maxOutputTokens=50, las respuestas fueron muy cortas, lo que no nos permitió obtener información suficiente para un informe médico. No fue útil para nuestro propósito.

### 3. Diferencia entre topP bajo y alto
#### Con topP bajo, las respuestas fueron más seguras. Con topP alto, las respuestas fueron más creativas y variables.

### 4. ¿Las respuestas con temperature=0 fueron idénticas? Implicancias para reproducibilidad
#### A mi (Catalina), el test me arrojó que no todas las respuestas fueron idénticas. En el tercer intento, el modelo puso las negritas (**) en otra posición (**Tejido conectivo** (o conjuntivo) en vez de **Tejido conectivo (o conjuntivo)**). Esto demuestra empíricamente que incluso con Temperature=0, las arquitecturas pueden tener ínfimas variabilidades. No son 100% deterministas.
#### En el caso de mi compañera (Melody), el test le arrojó que todas las respuestas fueron idénticas, el cual es el comportamiento que esperamos. Las ¨negritas¨ no variaron. 

### 5. Hiperparámetros ideales para un chatbot médico. Justificá.
#### Temperature: 0.0 → para mayor precisión (menos creatividad).
#### TopP: bajo → para mayor seguridad (menos opciones).
#### TopK: bajo → para mayor seguridad (menos opciones).
#### MaxOutputTokens: alto → para que la información sea suficiente y no se corte a la mitad de la respuesta


---

## Ejercicio 3: Prompt Engineering

### 1. Ranking de técnicas (peor a mejor) con justificación
#### La peor es zero-shot, ya que no damos contexto ni ejemplos, por lo que el modelo responde de manera general.
#### Luego sigue few-shot, ya que damos ejemplos pero no indicamos el rol bajo el cual analizar el caso, ni podemos saber en qué momento de su lógica se confundió si es que hay un error.
#### Luego sigue chain-of-thought, ya que indicamos el formato de salida pero no indicamos el rol que debe asumir el modelo.
#### La mejor es role + constraints, ya que indicamos el formato de salida y el rol que debe asumir el modelo.


### 2. ¿La respuesta JSON fue clínicamente correcta? Ventajas del output estructurado
#### Sí, el JSON arrojó como diagnóstico principal: "Anemia megaloblástica por deficiencia de vitamina B12" con certeza "alta", lo cual es clínicamente correcto acorde a los laboratorios. La principal ventaja es que permite obtener respuestas precisas y consistentes, además de facilitar el análisis de los resultados.

### 3. ¿El chain-of-thought cambió el diagnóstico o solo el razonamiento?
#### Solo cambió el razonamiento. El modelo ya había acertado el diagnóstico de anemia megaloblástica por deficiencia de B12. Sin embargo, el "cómo" en medicina es fundamental: detalla la cadena de pensamiento paso a paso y permite auditar la lógica clínica del modelo. Esto aporta "explicabilidad"  y vuelve al sistema  más seguro y confiable.

### 4. ¿Encontraste información incorrecta presentada con confianza? ¿Cómo mitigarlo?
#### No encontramos información incorrecta presentada con confianza. Sin embargo, es importante mencionar que en un contexto real, los LLMs pueden alucinar información, por lo que es fundamental validar sus respuestas con fuentes confiables. Para mitigar esto, se deberia indicar bien el System Prompt el tipo de respuesta que se busca y también se deberia utilizar una combinacion de tecnicas de Prompt Engineering para asegurar que el modelo responda de manera precisa y consistente.

### 5. Tu diseño ideal de asistente diagnóstico
#### Un asistente ideal combinaría todas las técnicas como por ejemplo: Role + Constraints, Chain-of-Thought y Few-Shot.  

---

## Reflexión Final

### ¿Qué aprendiste que no esperabas?
#### Que incluso forzando el modelo a ser determinista ( con Temperature = 0), el modelo no da respuestas 100% idénticas en todas sus corridas. Además, ante un límite de palabras corto (maxOutputTokens bajo), se observó que la IA prefiere "cortarse a la mitad de una sílaba" antes que resumir el texto. También, nos sorprendió lo eficiente que puede ser un "System Prompt" para obligarla a no alucinar.

### ¿Qué riesgos ves en el uso de LLMs en medicina?
#### Uno le los riesgos son las alucinaciones "creíbles". El modelo puede inventarse esquemas de medicación o citar evidencia con total confianza, lo que puede llevar a un error grave si el profesional se apoya ciegamente en él. Otro riesgo es la privacidad de los datos, no se deberia incluir datos personales de los pacientes en los prompts.

### ¿Qué oportunidades ves para tu área de especialización?
#### Relacionado a la ingeniería biomédica, los LLMs permiten convertir historias clínicas en bases de datos para ser utilizadas en hospitales, crear tutores interactivos que ayuden a filtrar pacientes y poder extraer marcadores clave, aumentando la eficiencia del personal y mejorando la calidad de atención al paciente.

