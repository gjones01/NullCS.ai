import { useEffect, useRef } from "react";
import * as THREE from "three";

const TARGET_FPS = 26;

export function ForensicBackground({ reducedMotion }: { reducedMotion: boolean }) {
  const mountRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (reducedMotion) return;
    const mount = mountRef.current;
    if (!mount) return;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.z = 28;

    const renderer = new THREE.WebGLRenderer({ antialias: false, alpha: true, powerPreference: "low-power" });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
    renderer.setSize(window.innerWidth, window.innerHeight);
    mount.appendChild(renderer.domElement);

    const count = 140;
    const positions = new Float32Array(count * 3);
    const labels: HTMLDivElement[] = [];

    for (let i = 0; i < count; i += 1) {
      positions[i * 3] = (Math.random() - 0.5) * 44;
      positions[i * 3 + 1] = (Math.random() - 0.5) * 28;
      positions[i * 3 + 2] = (Math.random() - 0.5) * 28;
      if (i % 18 === 0) {
        const el = document.createElement("div");
        el.className = "forensic-label";
        el.textContent = `${Math.floor(Math.random() * 9000) + 1000}`;
        mount.appendChild(el);
        labels.push(el);
      }
    }

    const pointGeo = new THREE.BufferGeometry();
    pointGeo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    const points = new THREE.Points(
      pointGeo,
      new THREE.PointsMaterial({ color: 0xb9f4ff, size: 0.14, transparent: true, opacity: 0.68 })
    );
    scene.add(points);

    const linePairs: number[] = [];
    for (let i = 0; i < count - 1; i += 1) {
      if (i % 3 === 0) linePairs.push(i, (i + 7) % count);
    }
    const linePos = new Float32Array(linePairs.length * 3);
    linePairs.forEach((idx, i) => {
      linePos[i * 3] = positions[idx * 3];
      linePos[i * 3 + 1] = positions[idx * 3 + 1];
      linePos[i * 3 + 2] = positions[idx * 3 + 2];
    });
    const lineGeo = new THREE.BufferGeometry();
    lineGeo.setAttribute("position", new THREE.BufferAttribute(linePos, 3));
    const lines = new THREE.LineSegments(
      lineGeo,
      new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.13 })
    );
    scene.add(lines);

    let raf = 0;
    let last = 0;
    const frameMs = 1000 / TARGET_FPS;

    const animate = (t: number) => {
      raf = requestAnimationFrame(animate);
      if (t - last < frameMs) return;
      last = t;

      points.rotation.y += 0.0016;
      points.rotation.x += 0.0008;
      lines.rotation.y -= 0.001;
      lines.rotation.z += 0.0004;

      labels.forEach((el, i) => {
        const alpha = 0.2 + 0.5 * Math.abs(Math.sin(t * 0.0005 + i));
        el.style.opacity = `${alpha.toFixed(3)}`;
        el.style.transform = `translate(${8 + ((i * 149) % window.innerWidth)}px, ${16 + ((i * 113) % window.innerHeight)}px)`;
      });

      renderer.render(scene, camera);
    };
    raf = requestAnimationFrame(animate);

    const onResize = () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    };
    window.addEventListener("resize", onResize);

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", onResize);
      labels.forEach((l) => l.remove());
      renderer.dispose();
      mount.innerHTML = "";
    };
  }, [reducedMotion]);

  return <div ref={mountRef} className="forensic-bg" aria-hidden />;
}
