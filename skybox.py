from texture import *


class MappingTextureCube:

    def __init__(self, tab_texture):
        self.glid = GL.glGenTextures(1)
        self.type = GL.GL_TEXTURE_CUBE_MAP
        try:
            for i in range(0, len(tab_texture)):
                tex = Image.open(tab_texture[i]).convert('RGBA')
                GL.glBindTexture(self.type, self.glid)
                GL.glTexImage2D(GL.GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL.GL_RGBA,
                                tex.width, tex.height, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, tex.tobytes())
                GL.glTexParameteri(GL.GL_TEXTURE_CUBE_MAP,
                                   GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
                GL.glTexParameteri(GL.GL_TEXTURE_CUBE_MAP,
                                   GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
                GL.glTexParameteri(GL.GL_TEXTURE_CUBE_MAP,
                                   GL.GL_TEXTURE_WRAP_R, GL.GL_CLAMP_TO_EDGE)
                GL.glTexParameteri(GL.GL_TEXTURE_CUBE_MAP,
                                   GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
                GL.glTexParameteri(GL.GL_TEXTURE_CUBE_MAP,
                                   GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)

                i += 1
        except Exception as e:
            print(e + "Can't load texture")

    # Delete texture from memory when object is deleted
    def __del__(self):
        GL.glDeleteTextures(self.glid)


class SkyBoxTexture(TexturedSky):

    def __init__(self, shader, tab_texture):

        # Face devant / derriere
        xpos_pos = ((-1, 1, -1), (-1, -1, -1),
                    (1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1))
        xneg_pos = ((-1, -1, 1), (-1, -1, -1),
                    (-1, 1, -1), (-1, 1, -1), (-1, 1, 1), (-1, -1, 1))
        ypos_pos = ((1, -1, -1), (1, -1, 1),
                    (1, 1, 1), (1, 1, 1), (1, 1, -1), (1, -1, -1))
        yneg_pos = ((-1, -1, 1), (-1, 1, 1),
                    (1, 1, 1), (1, 1, 1), (1, -1, 1), (-1, -1, 1))
        zpos_pos = ((-1, 1, -1), (1, 1, -1),
                    (1, 1, 1), (1, 1, 1), (-1, 1, 1), (-1, 1, -1))
        zneg_pos = ((-1, -1, -1), (-1, -1, 1),
                    (1, -1, -1), (1, -1, -1), (-1, -1, 1), (1, -1, 1))

        coords = xpos_pos+xneg_pos+ypos_pos+yneg_pos + zpos_pos + zneg_pos
        mesh = Mesh(shader, attributes=dict(position=coords))

        texture = MappingTextureCube(
            tab_texture)
        super().__init__(mesh, skybox=texture)
