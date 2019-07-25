uniform mat4 uTransformM;
uniform mat4 uOrientationM;
uniform vec2 ratios;
attribute vec2 aPosition;

varying vec2 vTextureCoord;

void main(){
	gl_Position = vec4(aPosition, 0.0, 1.0);
	vTextureCoord = (uTransformM * ((uOrientationM * gl_Position + 1.0)*0.5)).xy;
	gl_Position.xy *= ratios;
}