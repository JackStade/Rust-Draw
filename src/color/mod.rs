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

// RGBA, 8 bits per channel
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

/// Generates a color from a hex code
pub fn from_hex(color: u32) -> Color8Bit {
    Color8Bit { color: [0; 4] }
}
