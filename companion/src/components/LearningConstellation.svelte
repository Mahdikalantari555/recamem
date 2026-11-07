<script lang="ts">
  import { onMount } from 'svelte';

  let host: HTMLDivElement;
  let available = $state(false);

  onMount(() => {
    let disposed = false;
    let cleanup = () => {};
    // Load only on the workspace view. WebGL is optional; the CSS illustration
    // remains visible if importing Three.js or creating a context fails.
    void import('three').then((THREE) => {
      if (disposed || !host) return;
      let renderer: import('three').WebGLRenderer;
      try {
        renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true, powerPreference: 'low-power' });
      } catch {
        return;
      }
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
      renderer.setClearColor(0x000000, 0);
      renderer.outputColorSpace = THREE.SRGBColorSpace;
      host.appendChild(renderer.domElement);
      const scene = new THREE.Scene();
      const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 100);
      camera.position.set(0, 0, 7.6);
      const world = new THREE.Group();
      scene.add(world);

      scene.add(new THREE.AmbientLight(0xe0fae4, 2.0));
      const key = new THREE.DirectionalLight(0xffffff, 3.2);
      key.position.set(4, 6, 7);
      scene.add(key);
      const fill = new THREE.DirectionalLight(0x84bba2, 1.8);
      fill.position.set(-5, -2, -3);
      scene.add(fill);

      // A small, abstract seed at the center, not a representation of weights.
      const core = new THREE.Mesh(
        new THREE.IcosahedronGeometry(0.77, 2),
        new THREE.MeshStandardMaterial({ color: 0xc4efc2, roughness: 0.32, metalness: 0.12, flatShading: true })
      );
      world.add(core);
      const shell = new THREE.Mesh(
        new THREE.IcosahedronGeometry(1.05, 1),
        new THREE.MeshBasicMaterial({ color: 0xd7f2d4, wireframe: true, transparent: true, opacity: 0.32 })
      );
      world.add(shell);

      const orbitMaterial = new THREE.LineBasicMaterial({ color: 0xb5ddbb, transparent: true, opacity: 0.38 });
      const rings: import('three').LineLoop[] = [];
      for (let i = 0; i < 3; i++) {
        const points: import('three').Vector3[] = [];
        for (let n = 0; n < 96; n++) {
          const angle = n / 96 * Math.PI * 2;
          points.push(new THREE.Vector3(Math.cos(angle) * (1.55 + i * 0.32), Math.sin(angle) * (1.55 + i * 0.32), 0));
        }
        const ring = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(points), orbitMaterial);
        ring.rotation.set(0.8 + i * 0.35, 0.5 + i * 0.55, i * 0.8);
        world.add(ring);
        rings.push(ring);
      }

      const nodeGeometry = new THREE.SphereGeometry(0.055, 12, 12);
      const nodeMaterials = [
        new THREE.MeshStandardMaterial({ color: 0xd7efb7, emissive: 0x476f43, emissiveIntensity: 0.25 }),
        new THREE.MeshStandardMaterial({ color: 0xb4dace, emissive: 0x426f60, emissiveIntensity: 0.25 }),
        new THREE.MeshStandardMaterial({ color: 0xf0d9a3, emissive: 0x6e6242, emissiveIntensity: 0.2 }),
      ];
      const nodes: import('three').Mesh[] = [];
      for (let i = 0; i < 12; i++) {
        const node = new THREE.Mesh(nodeGeometry, nodeMaterials[i % nodeMaterials.length]);
        const angle = (i * 2.399963); // deterministic distribution
        const y = 1 - (i / 11) * 2;
        const radius = Math.sqrt(Math.max(0, 1 - y * y));
        node.position.set(Math.cos(angle) * radius * 2.05, y * 2.05, Math.sin(angle) * radius * 2.05);
        node.scale.setScalar(i % 3 === 0 ? 1.45 : 1);
        world.add(node);
        nodes.push(node);
      }
      const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
      let frame = 0;
      let lastTime = 0;
      let phase = 0;
      let lastDraw = 0;
      const render = (time = 0) => {
        frame = 0;
        if (disposed || document.hidden) return;
        // Keep the decorative scene lightweight while CPU training is running.
        if (!reducedMotion.matches && lastDraw && time && time - lastDraw < 33) {
          frame = requestAnimationFrame(render);
          return;
        }
        lastDraw = time;
        if (!reducedMotion.matches) {
          if (lastTime) phase += Math.min((time - lastTime) / 1000, 0.05);
          world.rotation.y = phase * 0.10;
          world.rotation.z = Math.sin(phase * 0.16) * 0.06;
          core.rotation.y = phase * 0.12;
        }
        lastTime = time;
        renderer.render(scene, camera);
        if (!reducedMotion.matches) frame = requestAnimationFrame(render);
      };
      const resize = () => {
        const width = Math.max(1, host.clientWidth);
        const height = Math.max(1, host.clientHeight);
        renderer.setSize(width, height, false);
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
        if (!frame) render();
      };
      const onVisibility = () => {
        if (document.hidden) { cancelAnimationFrame(frame); frame = 0; lastTime = 0; lastDraw = 0; }
        else if (!frame) render();
      };
      const onMotion = () => {
        cancelAnimationFrame(frame); frame = 0; lastTime = 0; lastDraw = 0;
        render();
      };
      const observer = new ResizeObserver(resize);
      observer.observe(host);
      document.addEventListener('visibilitychange', onVisibility);
      reducedMotion.addEventListener('change', onMotion);
      available = true;
      resize();
      cleanup = () => {
        cancelAnimationFrame(frame);
        observer.disconnect();
        document.removeEventListener('visibilitychange', onVisibility);
        reducedMotion.removeEventListener('change', onMotion);
        core.geometry.dispose(); (core.material as import('three').Material).dispose();
        shell.geometry.dispose(); (shell.material as import('three').Material).dispose();
        rings.forEach((ring) => ring.geometry.dispose());
        orbitMaterial.dispose(); nodeGeometry.dispose(); nodeMaterials.forEach((material) => material.dispose());
        renderer.dispose(); renderer.domElement.remove();
      };
    }).catch(() => { /* CSS fallback is decorative and remains usable. */ });
    return () => { disposed = true; cleanup(); };
  });
</script>

<div class:ready={available} class="constellation" bind:this={host} aria-hidden="true">
  <div class="constellation-fallback"><span>✳</span></div>
</div>

<style>
  .constellation { position: absolute; inset: 0; overflow: hidden; }
  .constellation :global(canvas) { display: block; width: 100%; height: 100%; }
  .constellation-fallback { position: absolute; left: 50%; top: 50%; width: 150px; height: 150px; transform: translate(-50%, -50%); border: 1px solid #b2d4b1; border-radius: 50%; display: grid; place-items: center; color: #d4efd3; font-size: 96px; }
  .constellation-fallback::before, .constellation-fallback::after { content: ''; position: absolute; inset: -28px; border: 1px solid #8ab39680; border-radius: 50%; transform: rotate(52deg) scaleY(.62); }
  .constellation-fallback::after { transform: rotate(-52deg) scaleY(.62); }
  .ready .constellation-fallback { display: none; }
</style>
