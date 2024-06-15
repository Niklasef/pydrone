import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
from pyrr import Matrix44, matrix44, Vector3
from CoordinateSystem import transform_to_global

def init(vertices, indices, static_vertices, static_indices, present_window=None):
    if not glfw.init():
        raise Exception("GLFW can't initialize")

    if present_window:
        window = present_window
    else:
        window = glfw.create_window(1820, 1024, "Simple Triangle", None, None)

    if not window:
        glfw.terminate()
        raise Exception("GLFW window can't be created")

    glfw.make_context_current(window)
    glEnable(GL_DEPTH_TEST)

    vertex_shader_src = """
    #version 330 core
    layout (location = 0) in vec3 aPos;
    layout (location = 1) in vec3 aNormal;
    layout (location = 2) in vec3 color;

    out vec3 Normal;
    out vec3 FragPos;
    out vec3 C;

    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;

    void main()
    {
        gl_Position = projection * view * model * vec4(aPos, 1.0);
        FragPos = vec3(model * vec4(aPos, 1.0));
        Normal = aNormal;
        C = color;
    }
    """

    fragment_shader_src = """
    #version 330 core
    out vec4 FragColor;

    in vec3 Normal;  
    in vec3 FragPos; 
    in vec3 C; 

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

        vec3 result = (ambient + diffuse) * C;
        FragColor = vec4(C, 1.0);
    }
    """

    shader = compileProgram(
        compileShader(vertex_shader_src, GL_VERTEX_SHADER), 
        compileShader(fragment_shader_src, GL_FRAGMENT_SHADER))
    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)
    
    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 36, ctypes.c_void_p(0))
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 36, ctypes.c_void_p(12))
    glEnableVertexAttribArray(2)
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 36, ctypes.c_void_p(24))



    EBO = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
    
   # Create Additional VAO, VBO, and EBO
    static_VAO = glGenVertexArrays(1)
    glBindVertexArray(static_VAO)

    static_VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, static_VBO)
    glBufferData(GL_ARRAY_BUFFER, static_vertices.nbytes, static_vertices, GL_STATIC_DRAW)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 36, ctypes.c_void_p(0))
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 36, ctypes.c_void_p(12))
    glEnableVertexAttribArray(2)
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 36, ctypes.c_void_p(24))

    static_EBO = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, static_EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, static_indices.nbytes, static_indices, GL_STATIC_DRAW)


    # Define vertices for the box frame
    box_vertices = np.array([
        0.7, -0.7,  # Bottom right
        0.7, -0.9,  # Top right
        0.5, -0.9,  # Top left
        0.5, -0.7   # Bottom left
    ], dtype=np.float32)


    box_indices = np.array([
        0, 1, 3,
        1, 2, 3
    ], dtype=np.uint32)

    # Box shaders
    box_vertex_shader_src = """
    #version 330 core
    layout (location = 0) in vec2 aPos;
    void main()
    {
        gl_Position = vec4(aPos, 0.0, 1.0);
    }
    """

    box_fragment_shader_src = """
    #version 330 core
    out vec4 FragColor;
    void main()
    {
        FragColor = vec4(1.0, 1.0, 1.0, 1.0); // White color
    }
    """

    box_shader = compileProgram(compileShader(box_vertex_shader_src, GL_VERTEX_SHADER), compileShader(box_fragment_shader_src, GL_FRAGMENT_SHADER))
    box_VAO = glGenVertexArrays(1)
    glBindVertexArray(box_VAO)

    box_VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, box_VBO)
    glBufferData(GL_ARRAY_BUFFER, box_vertices.nbytes, box_vertices, GL_STATIC_DRAW)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, ctypes.c_void_p(0))

    box_EBO = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, box_EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, box_indices.nbytes, box_indices, GL_STATIC_DRAW)



    box_vertices_two = np.array([
        -0.5, -0.7,  # Bottom right
        -0.5, -0.9,  # Top right
        -0.7, -0.9,  # Top left
        -0.7, -0.7   # Bottom left
    ], dtype=np.float32)
    box_VAO_two = glGenVertexArrays(1)
    glBindVertexArray(box_VAO_two)    
    box_VBO_two = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, box_VBO_two)
    glBufferData(GL_ARRAY_BUFFER, box_vertices_two.nbytes, box_vertices_two, GL_STATIC_DRAW)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, ctypes.c_void_p(0))

    box_EBO_two = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, box_EBO_two)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, box_indices.nbytes, box_indices, GL_STATIC_DRAW)



    # Dot shaders
    dot_vertex_shader_src = """
    #version 330 core
    layout (location = 0) in vec2 aPos;
    void main()
    {
        gl_Position = vec4(aPos, 0.0, 1.0);
    }
    """

    dot_fragment_shader_src = """
    #version 330 core
    out vec4 FragColor;
    void main()
    {
        FragColor = vec4(1.0, 0.0, 0.0, 1.0); // Red color
    }
    """

    dot_shader = compileProgram(compileShader(dot_vertex_shader_src, GL_VERTEX_SHADER), compileShader(dot_fragment_shader_src, GL_FRAGMENT_SHADER))
    
    dot_VAO = glGenVertexArrays(1)
    glBindVertexArray(dot_VAO)
    dot_VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, dot_VBO)
    dot_vertices = np.array([0.0, 0.0], dtype=np.float32)  # Initial position of the dot
    glBufferData(GL_ARRAY_BUFFER, dot_vertices.nbytes, dot_vertices, GL_STATIC_DRAW)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, ctypes.c_void_p(0))

    # Return dot_shader and dot_VAO along with other objects
    return window,shader, VAO,box_shader, box_VAO, box_VAO_two,dot_shader, dot_VAO, dot_VBO, static_VAO

def render(window, shader, VAO, indices, cam_y, cam_z, rotation, translation, box_shader, box_VAO, box_VAO_two, dot_shader, dot_VAO, dot_VBO, dot_x, dot_y, dot_two_x, dot_two_y, static_VAO, static_indices):
    glfw.poll_events()
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)  # Clear the depth buffer bit
    glUseProgram(shader)
    
    # Set the uniform values
    glUniform3f(glGetUniformLocation(shader, "lightPos"), 0.4, 0.4, -0.4)
    glUniform3f(glGetUniformLocation(shader, "viewPos"), 0.0, 0.0, 1.0)
    glUniform3f(glGetUniformLocation(shader, "lightColor"), 0.0, 1.0, 1.0)
    glUniform3f(glGetUniformLocation(shader, "objectColor"), 1.0, 0.5, 0.31)
    
    projection = matrix44.create_perspective_projection_matrix(45.0, 1820.0 / 1024.0, 0.1, 100.0)
    view = matrix44.create_from_translation(Vector3([0.0, cam_y, cam_z]))
    
    model = Matrix44.identity()  # Add a model matrix
    model = matrix44.multiply(model, rotation)  # Apply the rotation
    model = matrix44.multiply(model, translation)  # Apply the translation    
    glUniformMatrix4fv(glGetUniformLocation(shader, "projection"), 1, GL_FALSE, projection)
    glUniformMatrix4fv(glGetUniformLocation(shader, "view"), 1, GL_FALSE, view)
    glUniformMatrix4fv(glGetUniformLocation(shader, "model"), 1, GL_FALSE, model)

    glBindVertexArray(VAO)
    glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)

    glUseProgram(shader)
    static_model = Matrix44.identity()  # No rotation and translation
    glUniformMatrix4fv(glGetUniformLocation(shader, "model"), 1, GL_FALSE, static_model)

    glBindVertexArray(static_VAO)
    glDrawElements(GL_TRIANGLES, len(static_indices), GL_UNSIGNED_INT, None)


    # Draw 2D box
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glUseProgram(box_shader)
    glBindVertexArray(box_VAO)
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
    glDisable(GL_BLEND)
    # Draw 2D box
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glUseProgram(box_shader)
    glBindVertexArray(box_VAO_two)
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
    glDisable(GL_BLEND)    

    # Disable depth testing for the red dot
    glDisable(GL_DEPTH_TEST)

    # Draw 2D dot
    glBindVertexArray(dot_VAO)
    glBindBuffer(GL_ARRAY_BUFFER, dot_VBO)
    dot_vertices = np.array([dot_x, dot_y], dtype=np.float32)  # Update the position
    glBufferSubData(GL_ARRAY_BUFFER, 0, dot_vertices.nbytes, dot_vertices)  # Update the vertex data
    glUseProgram(dot_shader)
    glPointSize(10)  # Adjust the size as needed
    glDrawArrays(GL_POINTS, 0, 1)

    # Draw 2D dot
    glBindVertexArray(dot_VAO)
    glBindBuffer(GL_ARRAY_BUFFER, dot_VBO)
    dot_vertices = np.array([dot_two_x-1.2, dot_two_y], dtype=np.float32)  # Update the position
    glBufferSubData(GL_ARRAY_BUFFER, 0, dot_vertices.nbytes, dot_vertices)  # Update the vertex data
    glUseProgram(dot_shader)
    glPointSize(10)  # Adjust the size as needed
    glDrawArrays(GL_POINTS, 0, 1)    


    # Re-enable depth testing if needed for other objects
    glEnable(GL_DEPTH_TEST)

    glfw.swap_buffers(window)

def vertices_indices(drone):
    vertices_list = []
    indices_list = []
    face_indices = [
        0, 1, 2, 2, 3, 0,    # Front face
        1, 5, 6, 6, 2, 1,    # Right face
        7, 6, 5, 5, 4, 7,    # Back face
        4, 0, 3, 3, 7, 4,    # Left face
        4, 5, 1, 1, 0, 4,    # Bottom face
        3, 2, 6, 6, 7, 3     # Top face
    ]
    
    i = 0

    for _, spatial_object in enumerate(drone.spatial_objects):
        corners = [
            spatial_object.body.shape.left_bottom_inner_corner,
            spatial_object.body.shape.right_bottom_inner_corner,
            spatial_object.body.shape.right_top_inner_corner,
            spatial_object.body.shape.left_top_inner_corner,
            spatial_object.body.shape.left_bottom_outer_corner,
            spatial_object.body.shape.right_bottom_outer_corner,
            spatial_object.body.shape.right_top_outer_corner,
            spatial_object.body.shape.left_top_outer_corner,
        ]
        for corner in corners:
            vertices_list.extend(
                transform_to_global(spatial_object.coordinateSystem, corner))
            vertices_list.extend([0.0, 0.0, -1.0])
            if i == 0:
                vertices_list.extend([1.0, 0.0, 0.0])  # Red for first part
            elif i == 1:
                vertices_list.extend([0.0, 1.0, 0.0])  # Green for second part
            elif i == 2:
                vertices_list.extend([1.0, 1.0, 1.0])  # Green for second part
            elif i == 3:
                vertices_list.extend([0.0, 1.0, 0.0])  # Green for second part

        indices_list.extend([index + i*8 for index in face_indices])
        i += 1

    vertices = np.array(vertices_list, dtype=np.float32)
    indices = np.array(indices_list, dtype=np.uint32)

    return vertices, indices

def static_vertices_indices(nav_points):
    vertices_list = []
    indices_list = []
    face_indices = [
        0, 1, 2, 2, 3, 0,    # Front face
        1, 5, 6, 6, 2, 1,    # Right face
        7, 6, 5, 5, 4, 7,    # Back face
        4, 0, 3, 3, 7, 4,    # Left face
        4, 5, 1, 1, 0, 4,    # Bottom face
        3, 2, 6, 6, 7, 3     # Top face
    ]
    
    i = 0
    size = 0.1

    for _, nav_point in enumerate(nav_points):
        corners = [
            np.array([-size,-size,-size] + nav_point.position),
            np.array([size,-size,-size] + nav_point.position),
            np.array([-size,size,-size] + nav_point.position),
            np.array([size,size,-size] + nav_point.position),
            np.array([-size,-size,size] + nav_point.position),
            np.array([size,-size,size] + nav_point.position),
            np.array([-size,size,size] + nav_point.position),
            np.array([size,size,size] + nav_point.position),
        ]
        for corner in corners:
            vertices_list.extend(
                transform_to_global(nav_point.coordinate_system, corner))
            vertices_list.extend([0.0, 0.0, -1.0]) # Normal
            vertices_list.extend([1.0, 1.0, 1.0])  # White

        indices_list.extend([index + i*8 for index in face_indices])
        i += 1

    vertices = np.array(vertices_list, dtype=np.float32)
    indices = np.array(indices_list, dtype=np.uint32)

    return vertices, indices

def window_active(window):
    return not glfw.window_should_close(window)

def end():
    glfw.terminate()
