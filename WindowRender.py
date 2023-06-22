import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
from pyrr import Matrix44, matrix44, Vector3


def init(vertices, indices):
    if not glfw.init():
        raise Exception("GLFW can't initialize")

    window = glfw.create_window(720, 720, "Simple Triangle", None, None)

    if not window:
        glfw.terminate()
        raise Exception("GLFW window can't be created")

    glfw.make_context_current(window)
    glEnable(GL_DEPTH_TEST)

    vertex_shader_src = """
    #version 330 core
    layout (location = 0) in vec3 aPos;
    layout (location = 1) in vec3 aNormal;

    out vec3 Normal;
    out vec3 FragPos;

    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;

    void main()
    {
        gl_Position = projection * view * model * vec4(aPos, 1.0);
        FragPos = vec3(model * vec4(aPos, 1.0));
        Normal = aNormal;
    }
    """

    fragment_shader_src = """
    #version 330 core
    out vec4 FragColor;

    in vec3 Normal;  
    in vec3 FragPos;  

    uniform vec3 lightPos; 
    uniform vec3 viewPos; 
    uniform vec3 lightColor;
    uniform vec3 objectColor;

    void main()
    {
        // ambient
        float ambientStrength = 0.1;
        vec3 ambient = ambientStrength * lightColor;

        // diffuse 
        vec3 norm = normalize(Normal);
        vec3 lightDir = normalize(lightPos - FragPos);
        float diff = max(dot(norm, lightDir), 0.0);
        vec3 diffuse = diff * lightColor;

        vec3 result = (ambient + diffuse) * objectColor;
        FragColor = vec4(result, 1.0);
    }
    """

    shader = compileProgram(compileShader(vertex_shader_src, GL_VERTEX_SHADER), compileShader(fragment_shader_src, GL_FRAGMENT_SHADER))
    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)
    
    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))

    
    EBO = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
    
    return window, shader, VAO

def render(window, shader, VAO, indices, camera_distance, rotation, translation):
    glfw.poll_events()
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)  # Clear the depth buffer bit
    glUseProgram(shader)
    
    # Set the uniform values
    glUniform3f(glGetUniformLocation(shader, "lightPos"), 0.4, 0.4, -0.4)
    glUniform3f(glGetUniformLocation(shader, "viewPos"), 0.0, 0.0, 1.0)
    glUniform3f(glGetUniformLocation(shader, "lightColor"), 0.0, 1.0, 1.0)
    glUniform3f(glGetUniformLocation(shader, "objectColor"), 1.0, 0.5, 0.31)
    
    projection = matrix44.create_perspective_projection_matrix(45.0, 720.0 / 720.0, 0.1, 100.0)
    view = matrix44.create_from_translation(Vector3([0.0, 0.0, camera_distance]))
    
    model = Matrix44.identity()  # Add a model matrix
    model = matrix44.multiply(model, rotation)  # Apply the rotation
    model = matrix44.multiply(model, translation)  # Apply the translation    
    glUniformMatrix4fv(glGetUniformLocation(shader, "projection"), 1, GL_FALSE, projection)
    glUniformMatrix4fv(glGetUniformLocation(shader, "view"), 1, GL_FALSE, view)
    glUniformMatrix4fv(glGetUniformLocation(shader, "model"), 1, GL_FALSE, model)

    glBindVertexArray(VAO)
    glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)
    glfw.swap_buffers(window)


def window_active(window):
    return not glfw.window_should_close(window)

def end():
    glfw.terminate()
