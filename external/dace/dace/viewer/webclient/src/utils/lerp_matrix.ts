// Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

/**
 * Returns a function taking a number from 0 to 1 which linearly interpolates
 * between two matrices. Uses the matrix interpolation algorithm for CSS
 * animations:
 * https://www.w3.org/TR/css-transforms-1/#decomposing-a-2d-matrix
 */
export function lerpMatrix(
    m1: DOMMatrix, m2: DOMMatrix
): (t: number) => DOMMatrix {
    function decompose(m: DOMMatrix): any {
        const scale = [
            Math.sqrt(m.a * m.a + m.b * m.b),
            Math.sqrt(m.c * m.c + m.d * m.d)
        ];

        const det = m.a * m.d - m.b * m.c;
        if (det < 0) {
            if (m.a < m.d)
                scale[0] = -scale[0];
            else
                scale[1] = -scale[1];
        }

        const row0x = m.a / (scale[0] || 1);
        const row0y = m.b / (scale[0] || 1);
        const row1x = m.c / (scale[1] || 1);
        const row1y = m.d / (scale[1] || 1);

        const skew11 = row0x * row0x - row0y * row1x;
        const skew12 = row0x * row0y - row0y * row1y;
        const skew21 = row0x * row1x - row0y * row0x;
        const skew22 = row0x * row1y - row0y * row0y;

        const angle = Math.atan2(m.b, m.a) * 180 / Math.PI;

        return {
            translate: [m.e, m.f],
            scale,
            skew11,
            skew12,
            skew21,
            skew22,
            angle,
        };
    }

    function lerpDecomposed(d1: any, d2: any, t: number): any {
        function lerp(a: number, b: number): number {
            return (b - a) * t + a;
        }

        let d1Angle = d1.angle || 360;
        let d2Angle = d2.angle || 360;
        let d1Scale = d1.scale;

        if ((d1.scale[0] < 0 && d2.scale[1] < 0) ||
            (d1.scale[1] < 0 && d2.scale[0] < 0)) {
            d1Scale = [-d1Scale[0], -d1Scale[1]];
            d1Angle += d1Angle < 0 ? 180 : -180;
        }

        if (Math.abs(d1Angle - d2Angle) > 180) {
            if (d1Angle > d2Angle) {
                d1Angle -= 360;
            } else {
                d2Angle -= 360;
            }
        }


        return {
            translate: [
                lerp(d1.translate[0], d2.translate[0]),
                lerp(d1.translate[1], d2.translate[1]),
            ],
            scale: [
                lerp(d1Scale[0], d2.scale[0]),
                lerp(d1Scale[1], d2.scale[1]),
            ],
            skew11: lerp(d1.skew11, d2.skew11),
            skew12: lerp(d1.skew12, d2.skew12),
            skew21: lerp(d1.skew21, d2.skew21),
            skew22: lerp(d1.skew22, d2.skew22),
            angle: lerp(d1Angle, d2Angle),
        };
    }

    function recompose(d: any): DOMMatrix {
        const matrix = document.createElementNS(
            'http://www.w3.org/2000/svg', 'svg'
        ).createSVGMatrix();
        matrix.a = d.skew11;
        matrix.b = d.skew12;
        matrix.c = d.skew21;
        matrix.d = d.skew22;
        matrix.e = d.translate[0] * d.skew11 + d.translate[1] * d.skew21;
        matrix.f = d.translate[0] * d.skew12 + d.translate[1] * d.skew22;
        return matrix.rotate(0, 0, d.angle * Math.PI / 180).scale(
            d.scale[0], d.scale[1]
        );
    }

    const d1 = decompose(m1);
    const d2 = decompose(m2);

    return (t: number) => recompose(lerpDecomposed(d1, d2, t));
}
