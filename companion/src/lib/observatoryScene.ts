import type { ObservatoryNode, Tone } from './observatory';
import { placeOnShell, shellRadius } from './observatory';

/**
 * Framework-free Three.js scene for the learning observatory.
 *
 * Rules this module keeps (see openspec/changes/three-dimensional-learning-studio):
 * - the scene renders *only* the nodes handed to it; it never invents objects;
 * - the render loop runs only while motion is on and the document is visible,
 *   otherwise rendering is on demand (pointer, resize, selection);
 * - WebGL is optional: `createObservatoryScene` resolves to `supported: false`
 *   instead of throwing, so the caller keeps the semantic 2D view;
 * - context loss stops the loop and reports it; restoration resumes safely;
 * - every geometry, material, listener and the renderer itself is released by
 *   `dispose()`, so repeated mounts do not leak.
 */

export interface SceneOptions {
  onSelect?: (id: string | null) => void;
  onContextLost?: () => void;
  onContextRestored?: () => void;
  motion?: boolean;
}

export interface SceneHandle {
  supported: boolean;
  reason?: string;
  setNodes(nodes: ObservatoryNode[]): void;
  setSelected(id: string | null): void;
  setMotion(motion: boolean): void;
  resize(): void;
  render(): void;
  dispose(): void;
}

const TONE_COLORS: Record<Tone, number> = {
  green: 0x7fb583,
  lavender: 0x9a8eb9,
  amber: 0xd0a45c,
  grey: 0xb3bdb2,
};

const TONE_EMISSIVE: Record<Tone, number> = {
  green: 0x2f5c36,
  lavender: 0x3c3355,
  amber: 0x6b5220,
  grey: 0x3a4238,
};

export async function createObservatoryScene(
  host: HTMLElement,
  options: SceneOptions = {},
): Promise<SceneHandle> {
  let THREE: typeof import('three');
  try {
    THREE = await import('three');
  } catch {
    return unsupported('Three.js could not be loaded.');
  }

  let renderer: import('three').WebGLRenderer;
  try {
    renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true, powerPreference: 'low-power' });
  } catch {
    return unsupported('A WebGL context could not be created.');
  }

  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
  renderer.setClearColor(0x000000, 0);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.domElement.setAttribute('aria-hidden', 'true');
  host.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 60);
  camera.position.set(0, 2.2, 10.4);
  camera.lookAt(0, 0, 0);

  const world = new THREE.Group();
  scene.add(world);

  scene.add(new THREE.AmbientLight(0xf2f8ef, 2.2));
  const keyLight = new THREE.DirectionalLight(0xffffff, 2.6);
  keyLight.position.set(5, 7, 8);
  scene.add(keyLight);
  const rimLight = new THREE.DirectionalLight(0x9dc3a4, 1.4);
  rimLight.position.set(-6, -3, -5);
  scene.add(rimLight);

  // -- static scaffolding (decoration, named as such in the UI) ---------- #
  const disposables: Array<{ dispose(): void }> = [];
  const track = <T extends { dispose(): void }>(item: T): T => {
    disposables.push(item);
    return item;
  };

  const coreGeometry = track(new THREE.IcosahedronGeometry(0.62, 2));
  const coreMaterial = track(new THREE.MeshStandardMaterial({
    color: 0xcfe6c6, roughness: 0.42, metalness: 0.08, flatShading: true,
  }));
  const core = new THREE.Mesh(coreGeometry, coreMaterial);
  world.add(core);

  const haloGeometry = track(new THREE.IcosahedronGeometry(0.86, 1));
  const haloMaterial = track(new THREE.MeshBasicMaterial({
    color: 0xd8eed2, wireframe: true, transparent: true, opacity: 0.3,
  }));
  const halo = new THREE.Mesh(haloGeometry, haloMaterial);
  world.add(halo);

  const ringMaterials: import('three').LineBasicMaterial[] = [];
  const rings: import('three').LineLoop[] = [];
  for (let shell = 0; shell < 3; shell++) {
    const radius = shellRadius(shell);
    const points: import('three').Vector3[] = [];
    for (let step = 0; step < 128; step++) {
      const angle = (step / 128) * Math.PI * 2;
      points.push(new THREE.Vector3(Math.cos(angle) * radius, Math.sin(angle) * radius, 0));
    }
    const material = track(new THREE.LineBasicMaterial({
      color: 0xbfd8b9, transparent: true, opacity: 0.26 - shell * 0.05,
    }));
    ringMaterials.push(material);
    const ring = new THREE.LineLoop(track(new THREE.BufferGeometry().setFromPoints(points)), material);
    ring.rotation.set(Math.PI / 2 - shell * 0.16, shell * 0.22, 0);
    world.add(ring);
    rings.push(ring);
  }

  // -- data nodes -------------------------------------------------------- #
  const nodeGeometry = track(new THREE.SphereGeometry(0.075, 14, 14));
  const toneMaterials = new Map<Tone, import('three').MeshStandardMaterial>();
  (Object.keys(TONE_COLORS) as Tone[]).forEach((tone) => {
    toneMaterials.set(tone, track(new THREE.MeshStandardMaterial({
      color: TONE_COLORS[tone],
      emissive: TONE_EMISSIVE[tone],
      emissiveIntensity: 0.5,
      roughness: 0.4,
      metalness: 0.05,
    })));
  });
  const selectedMaterial = track(new THREE.MeshStandardMaterial({
    color: 0xffffff, emissive: 0x9ed7a4, emissiveIntensity: 1.1, roughness: 0.2,
  }));

  const nodeGroup = new THREE.Group();
  world.add(nodeGroup);
  let meshes: import('three').Mesh[] = [];
  let selectedId: string | null = null;

  const clearNodes = () => {
    for (const mesh of meshes) nodeGroup.remove(mesh);
    meshes = [];
  };

  const setNodes = (nodes: ObservatoryNode[]) => {
    clearNodes();
    const perShell = new Map<number, ObservatoryNode[]>();
    for (const node of nodes) {
      const list = perShell.get(node.shell) ?? [];
      list.push(node);
      perShell.set(node.shell, list);
    }
    for (const [shell, list] of perShell) {
      list.forEach((node, index) => {
        const position = placeOnShell(index, list.length, shell);
        const mesh = new THREE.Mesh(nodeGeometry, toneMaterials.get(node.tone) ?? toneMaterials.get('grey')!);
        mesh.position.set(position.x, position.y, position.z);
        mesh.scale.setScalar(node.size);
        mesh.userData.nodeId = node.id;
        mesh.userData.nodeTitle = node.title;
        mesh.userData.tone = node.tone;
        mesh.userData.baseScale = node.size;
        meshes.push(mesh);
        nodeGroup.add(mesh);
      });
    }
    applySelection();
    render();
  };

  const applySelection = () => {
    for (const mesh of meshes) {
      const isSelected = mesh.userData.nodeId === selectedId;
      const baseTone = (mesh.userData.tone as Tone | undefined) ?? 'grey';
      mesh.material = isSelected ? selectedMaterial : toneMaterials.get(baseTone) ?? selectedMaterial;
      const base = (mesh.userData.baseScale as number) ?? 1;
      mesh.scale.setScalar(isSelected ? base * 1.9 : base);
    }
  };

  const setSelected = (id: string | null) => {
    selectedId = id;
    applySelection();
    render();
  };

  // -- interaction ------------------------------------------------------- #
  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();
  let dragging = false;
  let dragMoved = 0;
  let lastPointer = { x: 0, y: 0 };

  const toPointer = (event: PointerEvent) => {
    const rect = renderer.domElement.getBoundingClientRect();
    pointer.x = ((event.clientX - rect.left) / Math.max(1, rect.width)) * 2 - 1;
    pointer.y = -((event.clientY - rect.top) / Math.max(1, rect.height)) * 2 + 1;
  };

  const pick = (event: PointerEvent): ObservatoryNode['id'] | null => {
    toPointer(event);
    raycaster.setFromCamera(pointer, camera);
    const hits = raycaster.intersectObjects(meshes, false);
    return hits.length ? (hits[0].object.userData.nodeId as string) : null;
  };

  const onPointerDown = (event: PointerEvent) => {
    dragging = true;
    dragMoved = 0;
    lastPointer = { x: event.clientX, y: event.clientY };
    renderer.domElement.setPointerCapture?.(event.pointerId);
  };

  const onPointerMove = (event: PointerEvent) => {
    if (dragging) {
      const dx = event.clientX - lastPointer.x;
      const dy = event.clientY - lastPointer.y;
      dragMoved += Math.abs(dx) + Math.abs(dy);
      lastPointer = { x: event.clientX, y: event.clientY };
      world.rotation.y += dx * 0.006;
      world.rotation.x = Math.max(-0.6, Math.min(0.6, world.rotation.x + dy * 0.004));
      render();
      return;
    }
    const hovered = pick(event);
    renderer.domElement.style.cursor = hovered ? 'pointer' : 'grab';
  };

  const onPointerUp = (event: PointerEvent) => {
    const wasDragging = dragging;
    dragging = false;
    if (!wasDragging || dragMoved < 6) {
      options.onSelect?.(pick(event));
    }
  };

  const onPointerLeave = () => {
    dragging = false;
  };

  renderer.domElement.style.cursor = 'grab';
  renderer.domElement.addEventListener('pointerdown', onPointerDown);
  renderer.domElement.addEventListener('pointermove', onPointerMove);
  renderer.domElement.addEventListener('pointerup', onPointerUp);
  renderer.domElement.addEventListener('pointerleave', onPointerLeave);

  // -- lifecycle --------------------------------------------------------- #
  let motion = options.motion ?? true;
  let frame = 0;
  let phase = 0;
  let lastTime = 0;
  let contextLost = false;

  const render = () => {
    if (contextLost) return;
    renderer.render(scene, camera);
  };

  const tick = (time = 0) => {
    frame = 0;
    if (contextLost || document.hidden) return;
    if (!motion) {
      render();
      return;
    }
    if (lastTime) phase += Math.min((time - lastTime) / 1000, 0.06);
    lastTime = time;
    world.rotation.y += 0.0025;
    core.rotation.y = phase * 0.22;
    halo.rotation.x = phase * 0.12;
    rings.forEach((ring, index) => {
      ring.rotation.z = phase * (0.05 + index * 0.02);
    });
    render();
    frame = requestAnimationFrame(tick);
  };

  const start = () => {
    if (frame || contextLost) return;
    lastTime = 0;
    frame = requestAnimationFrame(tick);
  };

  const stop = () => {
    if (frame) cancelAnimationFrame(frame);
    frame = 0;
  };

  const setMotion = (value: boolean) => {
    motion = value;
    if (motion) start();
    else { stop(); render(); }
  };

  const resize = () => {
    const width = Math.max(1, host.clientWidth);
    const height = Math.max(1, host.clientHeight);
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    render();
  };

  const onVisibility = () => {
    if (document.hidden) stop();
    else if (motion) start();
    else render();
  };

  const onContextLostEvent = (event: Event) => {
    event.preventDefault();
    contextLost = true;
    stop();
    options.onContextLost?.();
  };

  const onContextRestoredEvent = () => {
    contextLost = false;
    resize();
    if (motion) start();
    options.onContextRestored?.();
  };

  const canvas = renderer.domElement;
  canvas.addEventListener('webglcontextlost', onContextLostEvent as EventListener, false);
  canvas.addEventListener('webglcontextrestored', onContextRestoredEvent, false);
  document.addEventListener('visibilitychange', onVisibility);
  const observer = new ResizeObserver(() => resize());
  observer.observe(host);

  resize();
  if (motion) start();

  const dispose = () => {
    stop();
    observer.disconnect();
    document.removeEventListener('visibilitychange', onVisibility);
    canvas.removeEventListener('webglcontextlost', onContextLostEvent as EventListener);
    canvas.removeEventListener('webglcontextrestored', onContextRestoredEvent);
    canvas.removeEventListener('pointerdown', onPointerDown);
    canvas.removeEventListener('pointermove', onPointerMove);
    canvas.removeEventListener('pointerup', onPointerUp);
    canvas.removeEventListener('pointerleave', onPointerLeave);
    clearNodes();
    for (const item of disposables) {
      try { item.dispose(); } catch { /* already released */ }
    }
    try { renderer.dispose(); } catch { /* context already gone */ }
    canvas.remove();
  };

  return {
    supported: true,
    setNodes,
    setSelected,
    setMotion,
    resize,
    render,
    dispose,
  };
}

function unsupported(reason: string): SceneHandle {
  return {
    supported: false,
    reason,
    setNodes() {},
    setSelected() {},
    setMotion() {},
    resize() {},
    render() {},
    dispose() {},
  };
}
