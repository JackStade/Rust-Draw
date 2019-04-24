pub trait Color {
    fn as_rgba(&self) -> [f32; 4];
}

/// Floating point rgba color
pub struct FloatRGBA {
    pub color: [f32; 4],
}

impl Color for FloatRGBA {
    fn as_rgba(&self) -> [f32; 4] {
        self.color
    }
}

// RGBA color, 8 bits per channel
#[repr(align(4))]
pub struct Color8Bit {
    pub color: [u8; 4],
}

impl Color for Color8Bit {
    fn as_rgba(&self) -> [f32; 4] {
        let c = self.color;
        [
            c[0] as f32 / 255.0,
            c[1] as f32 / 255.0,
            c[2] as f32 / 255.0,
            c[3] as f32 / 255.0,
        ]
    }
}

pub fn byte_color(r: u8, g: u8, b: u8) -> Color8Bit {
    Color8Bit {
        color: [r, g, b, 255],
    }
}

pub fn byte_color_with_alpha(r: u8, g: u8, b: u8, a: u8) -> Color8Bit {
    Color8Bit {
        color: [r, g, b, a],
    }
}

pub fn gray(val: u8) -> Color8Bit {
    Color8Bit {
        color: [val, val, val, 255],
    }
}

/// Generates a color from a hex code.
pub fn from_hex_with_alpha(color: u32) -> Color8Bit {
    Color8Bit {
        color: [
            ((color >> 16) & 0xFF) as u8,
            ((color >> 8) & 0xFF) as u8,
            (color & 0xFF) as u8,
            ((color >> 24) & 0xFF) as u8,
        ],
    }
}

/// Generates a color from a hex code.
pub fn from_hex(color: u32) -> Color8Bit {
    Color8Bit {
        color: [
            ((color >> 16) & 0xFF) as u8,
            ((color >> 8) & 0xFF) as u8,
            (color & 0xFF) as u8,
            255,
        ],
    }
}
