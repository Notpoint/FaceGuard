import AppKit
import Carbon
import AVFoundation
import Vision
import CoreImage
import ImageIO
import UniformTypeIdentifiers

func areaVolume(_ area: Double, low: Double, high: Double, minimum: Double, maximum: Double) -> Double {
    let t = min(1, max(0, (area - low) / max(0.000001, high - low)))
    return minimum + t * (maximum - minimum)
}

func sameFace(_ a: CGRect, _ b: CGRect) -> Bool {
    let intersection = a.intersection(b)
    guard !intersection.isNull else { return false }
    let overlap = intersection.width * intersection.height
    let smaller = min(a.width * a.height, b.width * b.height)
    let larger = max(a.width * a.height, b.width * b.height)
    // Merge shifted/expanded detections of the same face, including expressions.
    // Keep very different scales separate so a real distant face is not erased.
    let union = a.width * a.height + b.width * b.height - overlap
    let comparableSize = larger > 0 && smaller / larger >= 0.25
    let closeCenters = abs(a.midX - b.midX) <= max(a.width, b.width) * 0.35 &&
                       abs(a.midY - b.midY) <= max(a.height, b.height) * 0.35
    return smaller > 0 && comparableSize && closeCenters &&
           (overlap / max(union, 0.000001) >= 0.28 || overlap / smaller >= 0.75)
}

func mergeFaces(_ boxes: [CGRect]) -> [CGRect] {
    var unique: [CGRect] = []
    for box in boxes where !unique.contains(where: { sameFace($0, box) }) {
        unique.append(box)
    }
    return unique
}

func fullFrameBox(_ box: CGRect, tile: CGRect) -> CGRect {
    CGRect(x: tile.minX + box.minX * tile.width, y: tile.minY + box.minY * tile.height,
           width: box.width * tile.width, height: box.height * tile.height)
}

// A crop can cut a known face into mouth/chin fragments that Vision calls a face.
// Only suppress crop-generated fragments of a full-frame detection; preserve
// independently detected full-frame faces and candidates outside the known face.
func isCroppedFaceFragment(_ candidate: CGRect, tile: CGRect, fullFaces: [CGRect]) -> Bool {
    let area = candidate.width * candidate.height
    guard area > 0 else { return true }
    return fullFaces.contains { face in
        let overlap = candidate.intersection(face)
        guard !overlap.isNull else { return false }
        let mostlyInside = overlap.width * overlap.height / area >= 0.8
        let centerInside = face.contains(CGPoint(x: candidate.midX, y: candidate.midY))
        // Allow slight numerical differences at the crop edge.
        let cutsKnownFace = !tile.insetBy(dx: -0.005, dy: -0.005).contains(face)
        return mostlyInside && centerInside && cutsKnownFace
    }
}

// Geometry only: no identity recognition or face templates.
func needsAlarm(widths: [Double]) -> Bool {
    widths.count >= 2
}

struct Detection {
    var faces: [CGRect]
    var elapsed: Double = 0
    var scans: Int = 0
    var shouldAlarm: Bool { faces.count >= 2 }
}

// Overlapping coarse/fine crops cover edges and make tiny faces occupy a larger
// fraction of the detector input. Sampling is spread across the frame each round.
func detectionTiles() -> [CGRect] {
    var tiles: [CGRect] = []
    for y in [0.0, 0.275, 0.55] {
        for x in [0.0, 0.275, 0.55] { tiles.append(CGRect(x: x, y: y, width: 0.45, height: 0.45)) }
    }
    for y in [0.0, 0.175, 0.35, 0.525, 0.7] {
        for x in [0.0, 0.175, 0.35, 0.525, 0.7] { tiles.append(CGRect(x: x, y: y, width: 0.30, height: 0.30)) }
    }
    return tiles
}

func overlappingKnownFaces(_ candidate: CGRect, known: [CGRect]) -> [CGRect] {
    let area = candidate.width * candidate.height
    guard area > 0 else { return [] }
    return known.filter {
        let overlap = candidate.intersection($0)
        return !overlap.isNull && overlap.width * overlap.height / area >= 0.35
    }
}

func contextSupportsAdditionalFace(_ candidate: CGRect, known: [CGRect], evidence: [CGRect]) -> Bool {
    evidence.contains { observed in sameFace(observed, candidate) && !known.contains(where: { existing in sameFace(observed, existing) }) }
}

// Owned by Camera and used only on its serial queue. Requests do not share state
// across independent tests or simultaneous callers; Vision receives the same inputs.
final class FaceDetector {
    private let faceRequest = VNDetectFaceRectanglesRequest()
    private let tileRequest = VNDetectFaceRectanglesRequest()
    private let regions = detectionTiles()
    private let verificationRequest = VNDetectFaceRectanglesRequest()
    private func verifyAdditionalFace(_ candidate: CGRect, known: [CGRect], image: CGImage) throws -> Bool {
        let overlaps = overlappingKnownFaces(candidate, known: known)
        guard !overlaps.isEmpty else { return true }
        let union = overlaps.reduce(candidate) { $0.union($1) }
        let region = union.insetBy(dx: -union.width * 0.25, dy: -union.height * 0.25)
            .intersection(CGRect(x: 0, y: 0, width: 1, height: 1))
        let pixels = CGRect(x: region.minX * Double(image.width), y: (1 - region.maxY) * Double(image.height),
                            width: region.width * Double(image.width), height: region.height * Double(image.height)).integral
        guard let crop = image.cropping(to: pixels) else { return true }
        let tile = CGRect(x: pixels.minX / Double(image.width), y: 1 - pixels.maxY / Double(image.height),
                          width: pixels.width / Double(image.width), height: pixels.height / Double(image.height))
        try VNImageRequestHandler(cgImage: crop).perform([verificationRequest])
        let evidence = (verificationRequest.results ?? []).filter { $0.confidence >= 0.45 }.map { fullFrameBox($0.boundingBox, tile: tile) }
        return contextSupportsAdditionalFace(candidate, known: overlaps, evidence: evidence)
    }
// Shared by camera and image tests: no screenshot-specific regions or thresholds.
func detect(_ image: CGImage, thorough: Bool, cursor: inout Int,
                 partial: ((Detection) -> Void)? = nil) throws -> Detection {
    let started = ProcessInfo.processInfo.systemUptime
    try VNImageRequestHandler(cgImage: image, orientation: .up).perform([faceRequest])
    var faces = mergeFaces((faceRequest.results ?? []).filter { $0.confidence >= 0.45 }.map { $0.boundingBox })
    let fullFaces = faces
    partial?(Detection(faces: faces))
    let tiles = regions
    var scans = 0
    let startCursor = cursor
    for offset in 0..<tiles.count {
        let index = (startCursor + offset * 13) % tiles.count
        let region = tiles[index]
        let x = region.minX, y = region.minY, side = region.width
        let pixelRect = CGRect(x: x * Double(image.width), y: (1 - y - side) * Double(image.height),
                               width: side * Double(image.width), height: side * Double(image.height)).integral
        if let crop = image.cropping(to: pixelRect) {
            let tile = CGRect(x: pixelRect.minX / Double(image.width), y: 1 - pixelRect.maxY / Double(image.height),
                              width: pixelRect.width / Double(image.width), height: pixelRect.height / Double(image.height))
            let request = tileRequest
            try VNImageRequestHandler(cgImage: crop).perform([request])
            for face in request.results ?? [] where face.confidence >= 0.45 {
                let b = face.boundingBox
                if (x > 0 && b.minX < 0.015) || (region.maxX < 0.999 && b.maxX > 0.985) ||
                   (y > 0 && b.minY < 0.015) || (region.maxY < 0.999 && b.maxY > 0.985) { continue }
                let candidate = fullFrameBox(b, tile: tile)
                if faces.contains(where: { sameFace(candidate, $0) }) { continue }
                if isCroppedFaceFragment(candidate, tile: tile, fullFaces: fullFaces) { continue }
                if !(try verifyAdditionalFace(candidate, known: faces, image: image)) { continue }
                faces.append(candidate)
            }
        }
        faces = mergeFaces(faces)
        partial?(Detection(faces: faces))
        scans += 1
        if !thorough && offset >= 5 && ProcessInfo.processInfo.systemUptime - started >= 0.22 {
            cursor = (index + 13) % tiles.count
            break
        }
    }
    return Detection(faces: mergeFaces(faces), elapsed: ProcessInfo.processInfo.systemUptime - started, scans: scans)
}

}

final class Camera: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    let session = AVCaptureSession()
    let queue = DispatchQueue(label: "FaceGuard.camera")
    var onResult: ((Detection, CGImage?) -> Void)?
    var onEarlyResult: ((Detection) -> Void)?
    var onError: ((String) -> Void)?
    let detector = FaceDetector()
    private var tileCursor = 0
    private var configured = false
    private var wanted = false
    private var lastFrame = 0.0
    private let imageContext = CIContext(options: [.cacheIntermediates: false])
    func start() {
        queue.async { self.wanted = true }
        AVCaptureDevice.requestAccess(for: .video) { granted in
            guard granted else {
                DispatchQueue.main.async { self.onError?("未获得摄像头权限。请到系统设置 → 隐私与安全性 → 摄像头，允许 FaceGuard，然后重启应用。") }
                return
            }
            self.queue.async {
                guard self.wanted else { return }
                do {
                    if !self.configured {
                        guard let device = AVCaptureDevice.default(for: .video) else {
                            throw NSError(domain: "Camera", code: 1, userInfo: [NSLocalizedDescriptionKey: "未找到摄像头"])
                        }
                        let input = try AVCaptureDeviceInput(device: device)
                        self.session.beginConfiguration()
                        self.session.sessionPreset = .high
                        let output = AVCaptureVideoDataOutput()
                        output.alwaysDiscardsLateVideoFrames = true
                        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
                        output.setSampleBufferDelegate(self, queue: self.queue)
                        guard self.session.canAddInput(input), self.session.canAddOutput(output) else {
                            self.session.commitConfiguration()
                            throw NSError(domain: "Camera", code: 2, userInfo: [NSLocalizedDescriptionKey: "无法连接摄像头"])
                        }
                        self.session.addInput(input)
                        self.session.addOutput(output)
                        for preset: AVCaptureSession.Preset in [.hd1920x1080, .hd1280x720, .high] {
                            if self.session.canSetSessionPreset(preset) {
                                self.session.sessionPreset = preset
                                break
                            }
                        }
                        self.session.commitConfiguration()
                        self.configured = true
                    }
                    self.session.startRunning()
                } catch {
                    let message = error.localizedDescription
                    DispatchQueue.main.async { self.onError?(message) }
                }
            }
        }
    }
    func stop() { queue.async { self.wanted = false; self.session.stopRunning() } }
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        let now = ProcessInfo.processInfo.systemUptime
        guard now - lastFrame >= 0.06 else { return }
        lastFrame = now
        guard let buffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        do {
            let input = CIImage(cvPixelBuffer: buffer)
            guard let image = imageContext.createCGImage(input, from: input.extent) else {
                throw NSError(domain: "FaceGuard", code: 1, userInfo: [NSLocalizedDescriptionKey: "无法读取摄像头图像"])
            }
            var emittedAlarm = false
            let result = try detector.detect(image, thorough: false, cursor: &tileCursor) { result in
                guard result.shouldAlarm, !emittedAlarm else { return }
                emittedAlarm = true
                DispatchQueue.main.async { self.onEarlyResult?(result) }
            }
            DispatchQueue.main.async { self.onResult?(result, image) }
        } catch {
            DispatchQueue.main.async { self.onError?("画面检测失败：\(error.localizedDescription)") }
        }
    }
}

// Draw the exact processed frame and its boxes in one bottom-left coordinate system.
// This avoids both live-preview lag and AV preview mirroring / layer conversions.
func previewRect(_ imageSize: CGSize, in bounds: CGRect) -> CGRect {
    let scale = min(bounds.width / imageSize.width, bounds.height / imageSize.height)
    let size = CGSize(width: imageSize.width * scale, height: imageSize.height * scale)
    return CGRect(x: bounds.midX - size.width / 2, y: bounds.midY - size.height / 2,
                  width: size.width, height: size.height)
}

final class Preview: NSView {
    var image: CGImage?
    var faces: [CGRect] = []
    var areaLow: Double = 0.1
    var areaHigh: Double = 3
    var showGuides = true
    var displayedSecondArea: Double? {
        let sorted = faces.sorted { $0.width * $0.height > $1.width * $1.height }
        return sorted.count > 1 ? Double(sorted[1].width * sorted[1].height) * 100 : nil
    }
    init(session: AVCaptureSession) { super.init(frame: .zero) }
    required init?(coder: NSCoder) { fatalError() }
    func drawFaces() { needsDisplay = true }
    override func draw(_ dirtyRect: NSRect) {
        NSColor.black.setFill()
        bounds.fill()
        guard let image = image, let context = NSGraphicsContext.current?.cgContext else { return }
        let rect = previewRect(CGSize(width: image.width, height: image.height), in: bounds)
        context.draw(image, in: rect)
        context.setStrokeColor(NSColor.systemGreen.cgColor)
        context.setLineWidth(2)
        let sorted = faces.sorted { $0.width * $0.height > $1.width * $1.height }
        func label(_ text: String, x: CGFloat, y: CGFloat, color: NSColor) {
            let attributes: [NSAttributedString.Key: Any] = [.font: NSFont.boldSystemFont(ofSize: 12), .foregroundColor: color, .backgroundColor: NSColor.black.withAlphaComponent(0.8)]
            let size = (text as NSString).size(withAttributes: attributes)
            (text as NSString).draw(at: CGPoint(x: max(rect.minX, min(x, rect.maxX - size.width)), y: max(rect.minY, min(y, rect.maxY - size.height))), withAttributes: attributes)
        }
        for (index, face) in sorted.enumerated() {
            let box = fullFrameBox(face, tile: rect)
            let color: NSColor = index == 1 ? .systemOrange : .systemGreen
            context.setStrokeColor(color.cgColor)
            context.setLineWidth(index == 1 ? 3 : 2)
            context.stroke(box)
            let area = Double(face.width * face.height) * 100
            label(String(format: "%@ %.3f%%", index == 1 ? "第二大脸" : "人脸", area), x: box.minX, y: box.maxY + 3, color: color)
        }
        if showGuides {
            for (index, area) in [areaLow, areaHigh].enumerated() {
                let total = max(0, min(100, area)) / 100 * rect.width * rect.height
                var w = sqrt(total), h = sqrt(total)
                if h > rect.height { h = rect.height; w = total / max(h, 0.001) }
                if w > rect.width { w = rect.width; h = total / max(w, 0.001) }
                let guide = CGRect(x: rect.maxX - w, y: rect.maxY - h, width: w, height: h)
                let color: NSColor = index == 0 ? .systemCyan : .systemYellow
                context.saveGState()
                context.setStrokeColor(color.cgColor)
                context.setLineWidth(1.5)
                context.setLineDash(phase: 0, lengths: [5, 4])
                context.stroke(guide)
                context.restoreGState()
                label(String(format: "%@参考 %.3f%%", index == 0 ? "下限" : "上限", area), x: rect.minX + 4, y: rect.maxY - CGFloat(index + 1) * 20, color: color)
            }
        }
    }
}

// Register one system shortcut; no keyboard polling or event tap is needed.
final class WindowHotKey {
    private var hotKey: EventHotKeyRef?
    private var handler: EventHandlerRef?
    private var held = false
    private let action: () -> Void
    private let identifier: UInt32
    private let signature: OSType = 0x4647484B // FGHK
    init(identifier: UInt32 = 1, action: @escaping () -> Void) { self.identifier = identifier; self.action = action }

    func register(keyCode: UInt32 = UInt32(kVK_Space), modifiers: UInt32 = UInt32(optionKey)) -> OSStatus {
        var eventTypes = [
            EventTypeSpec(eventClass: OSType(kEventClassKeyboard), eventKind: UInt32(kEventHotKeyPressed)),
            EventTypeSpec(eventClass: OSType(kEventClassKeyboard), eventKind: UInt32(kEventHotKeyReleased))
        ]
        let installed = eventTypes.withUnsafeMutableBufferPointer { types in
            InstallEventHandler(GetApplicationEventTarget(), { _, event, context in
                guard let event = event, let context = context else { return OSStatus(eventNotHandledErr) }
                let owner = Unmanaged<WindowHotKey>.fromOpaque(context).takeUnretainedValue()
                var identifier = EventHotKeyID()
                let result = GetEventParameter(event, EventParamName(kEventParamDirectObject), EventParamType(typeEventHotKeyID), nil,
                                               MemoryLayout<EventHotKeyID>.size, nil, &identifier)
                guard result == noErr, identifier.signature == owner.signature, identifier.id == owner.identifier else {
                    return OSStatus(eventNotHandledErr)
                }
                if GetEventKind(event) == UInt32(kEventHotKeyReleased) {
                    owner.held = false
                } else if !owner.held {
                    owner.held = true
                    // The application event target dispatches this handler on the main thread.
                    owner.action()
                }
                return noErr
            }, types.count, types.baseAddress, Unmanaged.passUnretained(self).toOpaque(), &handler)
        }
        guard installed == noErr else { return installed }
        let identifier = EventHotKeyID(signature: signature, id: identifier)
        let result = RegisterEventHotKey(keyCode, modifiers, identifier,
                                         GetApplicationEventTarget(), OptionBits(kEventHotKeyExclusive), &hotKey)
        if result != noErr { unregister() }
        return result
    }
    func unregister() {
        if let hotKey = hotKey { UnregisterEventHotKey(hotKey) }
        if let handler = handler { RemoveEventHandler(handler) }
        hotKey = nil
        handler = nil
        held = false
    }
    deinit { unregister() }
}

func shouldHideWindow(visible: Bool, miniaturized: Bool, active: Bool) -> Bool {
    visible && !miniaturized && active
}

func shouldEnforceApp(enabled: Bool, monitoring: Bool, alarm: Bool, hasTarget: Bool) -> Bool {
    enabled && monitoring && alarm && hasTarget
}

// Foreground assistance only; never records or restores the previous application.
final class ForegroundProtection {
    var onFailure: ((String) -> Void)?
    private var target: URL?
    private var timer: Timer?
    private var active = false
    private var opening = false
    private var generation = 0
    func setTarget(_ url: URL?) {
        setActive(false)
        target = url
    }
    func setActive(_ enabled: Bool) {
        guard active != enabled else { return }
        active = enabled
        generation += 1
        opening = false
        timer?.invalidate(); timer = nil
        if enabled {
            enforce()
            if active {
                timer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] _ in self?.enforce() }
            }
        }
    }
    private func enforce() {
        guard active, !opening, let target = target else { return }
        if let running = NSWorkspace.shared.runningApplications.first(where: { $0.bundleURL?.standardizedFileURL == target.standardizedFileURL && !$0.isTerminated }) {
            if NSWorkspace.shared.frontmostApplication?.processIdentifier != running.processIdentifier {
                running.unhide()
                _ = running.activate(options: [.activateIgnoringOtherApps])
            }
            return
        }
        guard FileManager.default.fileExists(atPath: target.path) else {
            fail("目标应用已移动或不存在，请重新选择。")
            return
        }
        opening = true
        let token = generation
        let configuration = NSWorkspace.OpenConfiguration()
        configuration.activates = false
        NSWorkspace.shared.openApplication(at: target, configuration: configuration) { [weak self] application, error in
            DispatchQueue.main.async {
                guard let self = self, self.active, self.generation == token else { return }
                self.opening = false
                guard let application = application, error == nil else {
                    self.fail("无法打开目标应用：\(error?.localizedDescription ?? "未知错误")")
                    return
                }
                application.unhide()
                _ = application.activate(options: [.activateIgnoringOtherApps])
            }
        }
    }
    private func fail(_ message: String) {
        setActive(false)
        onFailure?(message)
    }
    deinit { timer?.invalidate() }
}

// A concrete document frame gives NSScrollView a real scroll range.
final class SettingsDocument: NSView {
    override var isFlipped: Bool { true }
    let stack: NSStackView
    private var widthRule: NSLayoutConstraint!
    private var updating = false
    init(stack: NSStackView) {
        self.stack = stack
        super.init(frame: NSRect(x: 0, y: 0, width: 640, height: 1))
        addSubview(stack)
        widthRule = stack.widthAnchor.constraint(equalToConstant: 592)
        NSLayoutConstraint.activate([
            stack.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 24),
            stack.topAnchor.constraint(equalTo: topAnchor, constant: 24), widthRule
        ])
    }
    required init?(coder: NSCoder) { fatalError() }
    func fit(width: CGFloat, minimumHeight: CGFloat) {
        guard !updating else { return }
        updating = true
        defer { updating = false }
        widthRule.constant = max(1, width - 48)
        for field in stack.arrangedSubviews.compactMap({ $0 as? NSTextField }) where field.lineBreakMode == .byWordWrapping {
            field.preferredMaxLayoutWidth = widthRule.constant
        }
        layoutSubtreeIfNeeded()
        let height = max(minimumHeight, ceil(stack.fittingSize.height) + 48)
        if frame.size != NSSize(width: width, height: height) { setFrameSize(NSSize(width: width, height: height)) }
    }
}
final class SettingsScrollView: NSScrollView {
    override func layout() {
        super.layout()
        (documentView as? SettingsDocument)?.fit(width: contentSize.width, minimumHeight: contentSize.height)
    }
}

final class App: NSObject, NSApplicationDelegate, NSWindowDelegate {
    let camera = Camera()
    var windowHotKey: WindowHotKey?
    var emergencyHotKey: WindowHotKey?
    let foregroundProtection = ForegroundProtection()
    var protectionEnabled = UserDefaults.standard.bool(forKey: "foregroundProtectionEnabled")
    var targetPath = UserDefaults.standard.string(forKey: "foregroundTargetPath")
    var choosingTarget = false
    var protectionToggle: NSButton!
    var targetLabel = NSTextField(labelWithString: "未选择目标应用")

    var shortcutStatus = noErr
    var statusItem: NSStatusItem!
    var window: NSWindow!
    var status = NSTextField(labelWithString: "点击开始，允许摄像头权限")
    var preview: Preview!
    var diagnostics = NSTextField(labelWithString: "预览显示检测的同一帧，绿框为已检出人脸")
    var sizes = NSTextField(labelWithString: "当前人脸数量：—")
    var volumeLabel = NSTextField(labelWithString: "")
    var volumeSlider: NSSlider!
    var startButton: NSButton!
    var toggleItem: NSMenuItem!
    var sound: NSSound?
    var previewVisible: Bool { window.isVisible && window.occlusionState.contains(.visible) && !window.isMiniaturized }
    var running = false
    var alarming = false
    var clearSince: Double?
    var lastResult = 0.0
    var alarmTimer: Timer?
    var watchdog: Timer?
    var areaLow = UserDefaults.standard.object(forKey: "areaLow") as? Double ?? 0.1
    var areaHigh = UserDefaults.standard.object(forKey: "areaHigh") as? Double ?? 3.0
    var minimumVolume = UserDefaults.standard.object(forKey: "minimumVolume") as? Double ?? 0.08
    var secondArea: Double? = nil
    var areaLowField: NSTextField!
    var areaHighField: NSTextField!
    var minVolumeSlider: NSSlider!
    var areaInfo = NSTextField(labelWithString: "第二大脸面积：—")
    var volume = UserDefaults.standard.object(forKey: "volume") as? Double ?? 0.18

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.accessory)
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        statusItem.button?.title = "◉ 护屏"
        let menu = NSMenu()
        menu.addItem(withTitle: "显示／隐藏窗口  ⌥空格", action: #selector(toggleWindow), keyEquivalent: "")
        toggleItem = menu.addItem(withTitle: "开始检测", action: #selector(toggle), keyEquivalent: "")
        menu.addItem(withTitle: "暂停检测并解除应用置前  ⌃⌥P", action: #selector(pause), keyEquivalent: "")
        menu.addItem(.separator())
        menu.addItem(withTitle: "退出", action: #selector(quit), keyEquivalent: "q")
        for item in menu.items { item.target = self }
        statusItem.menu = menu
        if let url = Bundle.main.url(forResource: "beep", withExtension: "wav") { sound = NSSound(contentsOf: url, byReference: false) }
        makeWindow()
        foregroundProtection.setTarget(targetPath.map { URL(fileURLWithPath: $0) })
        foregroundProtection.onFailure = { [weak self] message in
            guard let self = self else { return }
            self.protectionEnabled = false
            self.protectionToggle.state = .off
            UserDefaults.standard.set(false, forKey: "foregroundProtectionEnabled")
            self.targetLabel.stringValue = message
        }
        let emergency = WindowHotKey(identifier: 2) { [weak self] in self?.pause() }
        if emergency.register(keyCode: UInt32(kVK_ANSI_P), modifiers: UInt32(controlKey | optionKey)) == noErr {
            emergencyHotKey = emergency
        }
        let shortcut = WindowHotKey { [weak self] in self?.toggleWindow() }
        shortcutStatus = shortcut.register()
        if shortcutStatus == noErr {
            windowHotKey = shortcut
        } else {
            let alert = NSAlert()
            alert.messageText = "⌥空格快捷键未能启用"
            alert.informativeText = "快捷键可能被其他应用占用（错误码 \(shortcutStatus)）。仍可通过菜单栏“护屏”打开或隐藏窗口。"
            alert.addButton(withTitle: "知道了")
            alert.runModal()
        }
        camera.onResult = { [weak self] result, image in
            guard let self = self, self.running else { return }
            self.received(result)
            if self.previewVisible { self.showDetection(result, image: image) }
        }
        camera.onEarlyResult = { [weak self] result in self?.received(result) }
        camera.onError = { [weak self] message in
            guard let self = self, self.running else { return }
            self.pause()
            self.status.stringValue = message
            self.showWindow()
        }
        watchdog = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { [weak self] _ in
            guard let self = self, self.running, ProcessInfo.processInfo.systemUptime - self.lastResult > 4 else { return }
            self.setAlarm(false)
            self.clearSince = nil
            self.status.stringValue = "等待摄像头画面… 若持续无画面，请暂停后重试。"
            self.statusItem.button?.title = "⚠ 护屏"
        }
        showWindow()
    }
    func makeWindow() {
        window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 680, height: min(720, (NSScreen.main?.visibleFrame.height ?? 800) - 80)), styleMask: [.titled, .closable, .miniaturizable, .resizable], backing: .buffered, defer: false)
        window.title = "护屏 · FaceGuard 1.16 · 可缩放与滚动设置"
        window.isReleasedWhenClosed = false
        window.delegate = self
        window.contentMinSize = NSSize(width: 600, height: 400)
        window.setFrameAutosaveName("FaceGuardSettingsWindow1.16")
        window.center()
        let stack = NSStackView()
        stack.orientation = .vertical
        stack.alignment = .leading
        stack.spacing = 10
        stack.translatesAutoresizingMaskIntoConstraints = false
        let scroll = SettingsScrollView()
        scroll.translatesAutoresizingMaskIntoConstraints = false
        scroll.hasVerticalScroller = true
        scroll.hasHorizontalScroller = false
        scroll.autohidesScrollers = false
        scroll.drawsBackground = false
        let document = SettingsDocument(stack: stack)
        scroll.documentView = document
        window.contentView!.addSubview(scroll)
        NSLayoutConstraint.activate([
            scroll.leadingAnchor.constraint(equalTo: window.contentView!.leadingAnchor),
            scroll.trailingAnchor.constraint(equalTo: window.contentView!.trailingAnchor),
            scroll.topAnchor.constraint(equalTo: window.contentView!.topAnchor)
        ])
        let title = NSTextField(labelWithString: "有其他人入镜，轻声提醒你")
        title.font = .boldSystemFont(ofSize: 23)
        stack.addArrangedSubview(title)
        let intro = NSTextField(wrappingLabelWithString: "只检测人脸，出现两张或更多脸立即报警。背影不计数；保留小脸增强和局部重复框过滤。仅本机处理，不保存画面。")
        stack.addArrangedSubview(intro)
        preview = Preview(session: camera.session)
        stack.addArrangedSubview(preview)
        preview.widthAnchor.constraint(equalTo: stack.widthAnchor).isActive = true
        preview.heightAnchor.constraint(equalToConstant: 270).isActive = true
        stack.addArrangedSubview(sizes)
        let shortcutLabel = NSTextField(labelWithString: "快捷键：⌥空格 · 显示／隐藏窗口，后台检测继续")
        shortcutLabel.font = .systemFont(ofSize: 12)
        shortcutLabel.textColor = .secondaryLabelColor
        stack.addArrangedSubview(shortcutLabel)
        diagnostics.font = .systemFont(ofSize: 12)
        stack.addArrangedSubview(diagnostics)
        volumeSlider = NSSlider(value: volume * 100, minValue: 1, maxValue: 100, target: self, action: #selector(settingsChanged))
        let audioRow = NSStackView(views: [volumeLabel, volumeSlider, NSButton(title: "试听", target: self, action: #selector(testSound))])
        audioRow.spacing = 12
        stack.addArrangedSubview(audioRow)
        areaLowField = NSTextField(string: String(format: "%.2f", areaLow))
        areaHighField = NSTextField(string: String(format: "%.2f", areaHigh))
        areaLowField.widthAnchor.constraint(equalToConstant: 70).isActive = true
        areaHighField.widthAnchor.constraint(equalToConstant: 70).isActive = true
        let areaRow = NSStackView(views: [NSTextField(labelWithString: "面积范围（占画面 %）："), areaLowField, NSTextField(labelWithString: "至"), areaHighField, NSButton(title: "应用", target: self, action: #selector(applyArea))])
        areaRow.spacing = 8
        stack.addArrangedSubview(areaRow)
        minVolumeSlider = NSSlider(value: minimumVolume * 100, minValue: 0, maxValue: 100, target: self, action: #selector(settingsChanged))
        stack.addArrangedSubview(NSStackView(views: [NSTextField(labelWithString: "最小音量"), minVolumeSlider]))
        stack.addArrangedSubview(areaInfo)
        let calibration = NSStackView(views: [NSButton(title: "当前第二大脸 → 下限", target: self, action: #selector(captureLow)), NSButton(title: "当前第二大脸 → 上限", target: self, action: #selector(captureHigh))])
        calibration.spacing = 10
        stack.addArrangedSubview(calibration)
        let guideToggle = NSButton(checkboxWithTitle: "在实时画面显示下限／上限参考框", target: self, action: #selector(toggleGuides(_:)))
        guideToggle.state = .on
        stack.addArrangedSubview(guideToggle)
        protectionToggle = NSButton(checkboxWithTitle: "多脸报警时持续置前指定应用（可选）", target: self, action: #selector(protectionChanged))
        protectionToggle.state = protectionEnabled ? .on : .off
        stack.addArrangedSubview(protectionToggle)
        targetLabel.stringValue = targetPath.map { "目标：" + FileManager.default.displayName(atPath: $0) } ?? "未选择目标应用"
        targetLabel.lineBreakMode = .byTruncatingMiddle
        let targetRow = NSStackView(views: [NSButton(title: "选择目标应用…", target: self, action: #selector(chooseTarget)), targetLabel])
        targetRow.spacing = 10
        stack.addArrangedSubview(targetRow)
        let protectionHelp = NSTextField(wrappingLabelWithString: "报警结束后停留在目标应用，不返回原页面。暂停检测可解除置前；菜单栏或快捷键 ⌃⌥P 可暂停。请先在目标应用打开希望展示的窗口。")
        protectionHelp.font = .systemFont(ofSize: 12)
        stack.addArrangedSubview(protectionHelp)
        status.maximumNumberOfLines = 3
        status.lineBreakMode = .byWordWrapping
        status.textColor = .secondaryLabelColor
        stack.addArrangedSubview(status)
        startButton = NSButton(title: "开始检测", target: self, action: #selector(toggle))
        startButton.bezelStyle = .rounded
        let buttons = NSStackView(views: [startButton, NSButton(title: "测试图片…", target: self, action: #selector(testImage)), NSButton(title: "隐藏到菜单栏", target: self, action: #selector(hideWindow))])
        buttons.spacing = 12
        // Main actions stay outside the scrolling settings and are always reachable.
        buttons.translatesAutoresizingMaskIntoConstraints = false
        window.contentView!.addSubview(buttons)
        NSLayoutConstraint.activate([
            buttons.leadingAnchor.constraint(equalTo: window.contentView!.leadingAnchor, constant: 24),
            buttons.trailingAnchor.constraint(lessThanOrEqualTo: window.contentView!.trailingAnchor, constant: -24),
            buttons.bottomAnchor.constraint(equalTo: window.contentView!.bottomAnchor, constant: -16),
            scroll.bottomAnchor.constraint(equalTo: buttons.topAnchor, constant: -16)
        ])
        for view in stack.arrangedSubviews {
            view.widthAnchor.constraint(lessThanOrEqualTo: stack.widthAnchor).isActive = true
        }
        targetLabel.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)
        settingsChanged()
        window.contentView!.layoutSubtreeIfNeeded()
        document.fit(width: scroll.contentSize.width, minimumHeight: scroll.contentSize.height)
        scroll.contentView.scroll(to: .zero)
        scroll.reflectScrolledClipView(scroll.contentView)
    }
    func syncProtection() {
        foregroundProtection.setActive(shouldEnforceApp(enabled: protectionEnabled && !choosingTarget, monitoring: running, alarm: alarming, hasTarget: targetPath != nil))
    }
    @objc func protectionChanged() {
        protectionEnabled = protectionToggle.state == .on
        if protectionEnabled && targetPath == nil {
            protectionEnabled = false
            protectionToggle.state = .off
            targetLabel.stringValue = "请先选择目标应用，再开启此项。"
        }
        UserDefaults.standard.set(protectionEnabled, forKey: "foregroundProtectionEnabled")
        syncProtection()
    }
    @objc func chooseTarget() {
        choosingTarget = true
        syncProtection()
        defer { choosingTarget = false; syncProtection() }
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.applicationBundle]
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.directoryURL = URL(fileURLWithPath: "/Applications")
        guard panel.runModal() == .OK, let url = panel.url else { return }
        guard let bundle = Bundle(url: url), let identifier = bundle.bundleIdentifier,
              identifier != Bundle.main.bundleIdentifier else {
            let alert = NSAlert(); alert.messageText = "请选择其他有效的应用程序"; alert.runModal(); return
        }
        targetPath = url.path
        foregroundProtection.setTarget(url)
        UserDefaults.standard.set(url.path, forKey: "foregroundTargetPath")
        targetLabel.stringValue = "目标：" + FileManager.default.displayName(atPath: url.path)
    }
    @objc func settingsChanged() {
        volume = volumeSlider.doubleValue / 100
        if let slider = minVolumeSlider {
            minimumVolume = min(slider.doubleValue / 100, volume)
            slider.doubleValue = minimumVolume * 100
            UserDefaults.standard.set(minimumVolume, forKey: "minimumVolume")
        }
        volumeLabel.stringValue = "最大音量 \(Int(volume * 100))%"
        updateAreaVolume()
        UserDefaults.standard.set(volume, forKey: "volume")
    }
    @objc func applyArea() {
        guard let low = Double(areaLowField.stringValue), let high = Double(areaHighField.stringValue),
              low.isFinite, high.isFinite, low >= 0, high > low, high <= 100 else {
            let alert = NSAlert()
            alert.messageText = "面积范围需要满足：0 ≤ 下限 < 上限 ≤ 100（%）"
            alert.runModal()
            return
        }
        areaLow = low; areaHigh = high
        UserDefaults.standard.set(low, forKey: "areaLow")
        UserDefaults.standard.set(high, forKey: "areaHigh")
        preview.areaLow = areaLow; preview.areaHigh = areaHigh; preview.drawFaces()
        updateAreaVolume()
    }
    @objc func toggleGuides(_ sender: NSButton) { preview.showGuides = sender.state == .on; preview.drawFaces() }
    @objc func captureLow() { captureArea(low: true) }
    @objc func captureHigh() { captureArea(low: false) }
    func captureArea(low: Bool) {
        guard let area = preview.displayedSecondArea else {
            let alert = NSAlert(); alert.messageText = "请先在预览中检测到至少两张脸"; alert.runModal(); return
        }
        // Capture the actual frame currently displayed, not a later partial detection.
        let value = String(format: "%.5f", area)
        if low { areaLowField.stringValue = value } else { areaHighField.stringValue = value }
        if low && area >= areaHigh {
            areaHighField.stringValue = String(format: "%.5f", min(100, area + max(0.01, area * 0.25)))
        } else if !low && area <= areaLow {
            areaLowField.stringValue = String(format: "%.5f", max(0, area * 0.5))
        }
        applyArea()
    }
    func updateAreaVolume() {
        let value = secondArea.map { areaVolume($0, low: areaLow, high: areaHigh, minimum: minimumVolume, maximum: volume) } ?? minimumVolume
        sound?.volume = Float(value)
        if previewVisible, let area = secondArea {
            areaInfo.stringValue = String(format: "第二大脸面积 %.3f%% · 当前音量 %.0f%%（最小 %.0f%%）", area, value * 100, minimumVolume * 100)
        } else if previewVisible {
            areaInfo.stringValue = "第二大脸面积：—（不足两张脸）"
        }
    }
    func showDetection(_ result: Detection, image: CGImage?) {
        preview.image = image
        preview.areaLow = areaLow; preview.areaHigh = areaHigh
        preview.faces = result.faces
        preview.drawFaces()
        if let image = image {
            diagnostics.stringValue = "\(image.width)×\(image.height) · \(Int(result.elapsed * 1000)) ms · 本轮 \(result.scans) 个分区 · 绿框：人脸"
        }
    }
    @objc func testImage() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.png, .jpeg, .heic]
        guard panel.runModal() == .OK, let url = panel.url else { return }
        pause()
        status.stringValue = "正在完整扫描图片…（仅分析，不发声）"
        camera.queue.async {
            do {
                guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
                      let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
                    throw NSError(domain: "FaceGuard", code: 2, userInfo: [NSLocalizedDescriptionKey: "无法打开图片"])
                }
                var cursor = 0
                let result = try self.camera.detector.detect(image, thorough: true, cursor: &cursor)
                DispatchQueue.main.async {
                    let areas = result.faces.map { Double($0.width * $0.height) * 100 }.sorted(by: >)
                    self.secondArea = areas.count >= 2 ? areas[1] : nil
                    self.updateAreaVolume()
                    self.showDetection(result, image: image)
                    self.sizes.stringValue = "检测到 \(result.faces.count) 张人脸"
                    self.status.stringValue = result.shouldAlarm ? "图片完整扫描：检出多脸（测试不发声）" : "图片完整扫描：未检出多脸"
                }
            } catch {
                DispatchQueue.main.async { self.status.stringValue = "图片测试失败：\(error.localizedDescription)" }
            }
        }
    }
    @objc func showWindow() {
        NSApp.unhide(nil)
        if let screen = window.screen ?? NSScreen.main {
            let visible = screen.visibleFrame
            var frame = window.frame
            frame.size.width = min(frame.width, visible.width)
            frame.size.height = min(frame.height, visible.height)
            frame.origin.x = max(visible.minX, min(frame.minX, visible.maxX - frame.width))
            frame.origin.y = max(visible.minY, min(frame.minY, visible.maxY - frame.height))
            window.setFrame(frame, display: true)
        }
        if window.isMiniaturized { window.deminiaturize(nil) }
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }
    @objc func toggleWindow() {
        if shouldHideWindow(visible: window.isVisible, miniaturized: window.isMiniaturized, active: NSApp.isActive) {
            hideWindow()
            NSApp.hide(nil)
        } else {
            showWindow()
        }
    }
    @objc func hideWindow() {
        window.orderOut(nil)
        preview.image = nil
        preview.faces = []
    }
    func windowShouldClose(_ sender: NSWindow) -> Bool { hideWindow(); return false }
    @objc func toggle() {
        if running { pause(); return }
        running = true
        lastResult = ProcessInfo.processInfo.systemUptime
        clearSince = nil
        startButton.title = "暂停检测"
        toggleItem.title = "暂停检测"
        status.stringValue = "正在连接摄像头…"
        camera.start()
    }
    @objc func pause() {
        running = false
        foregroundProtection.setActive(false)
        secondArea = nil
        camera.stop()
        setAlarm(false)
        startButton.title = "开始检测"
        toggleItem.title = "开始检测"
        status.stringValue = "已暂停，摄像头关闭"
        preview.image = nil
        preview.faces = []
        preview.drawFaces()
        sizes.stringValue = "当前人脸数量：—"
        statusItem.button?.title = "◉ 护屏 · 暂停"
    }
    func received(_ result: Detection) {
        guard running else { return }
        let now = ProcessInfo.processInfo.systemUptime
        lastResult = now
        let areas = result.faces.map { Double($0.width * $0.height) * 100 }.sorted(by: >)
        secondArea = areas.count >= 2 ? areas[1] : nil
        updateAreaVolume()
        if previewVisible { sizes.stringValue = "检测到 \(result.faces.count) 张人脸" }
        if result.shouldAlarm {
            clearSince = nil
            setAlarm(true)
            if previewVisible { status.stringValue = "检测到多张人脸 · 正在连续提醒" }
        } else {
            if clearSince == nil { clearSince = now }
            if now - clearSince! >= 0.65 { setAlarm(false) }
            if previewVisible { status.stringValue = alarming ? "等待画面恢复稳定…" : "检测中 · 未检测到多人" }
        }
        let title = alarming ? "⚠ 护屏" : "◉ 护屏"
        if statusItem.button?.title != title { statusItem.button?.title = title }
    }
    func setAlarm(_ enabled: Bool) {
        guard alarming != enabled else { return }
        alarming = enabled
        syncProtection()
        alarmTimer?.invalidate()
        alarmTimer = nil
        sound?.stop()
        if enabled {
            testSound()
            alarmTimer = Timer.scheduledTimer(withTimeInterval: 0.65, repeats: true) { [weak self] _ in self?.testSound() }
        }
    }
    @objc func testSound() { sound?.stop(); updateAreaVolume(); sound?.play() }
    func applicationWillTerminate(_ notification: Notification) { foregroundProtection.setActive(false); windowHotKey?.unregister(); emergencyHotKey?.unregister() }
    @objc func quit() { pause(); NSApp.terminate(nil) }
}

if CommandLine.arguments.contains("--self-test") {
    assert(!shouldEnforceApp(enabled: false, monitoring: true, alarm: true, hasTarget: true))
    assert(!shouldEnforceApp(enabled: true, monitoring: false, alarm: true, hasTarget: true))
    assert(!shouldEnforceApp(enabled: true, monitoring: true, alarm: false, hasTarget: true))
    assert(!shouldEnforceApp(enabled: true, monitoring: true, alarm: true, hasTarget: false))
    assert(shouldEnforceApp(enabled: true, monitoring: true, alarm: true, hasTarget: true))
    let knownFace = CGRect(x: 0.3, y: 0.3, width: 0.3, height: 0.4)
    let mouthFragment = CGRect(x: 0.39, y: 0.27, width: 0.12, height: 0.15)
    let independent = CGRect(x: 0.85, y: 0.2, width: 0.03, height: 0.05)
    assert(overlappingKnownFaces(mouthFragment, known: [knownFace]).count == 1)
    assert(overlappingKnownFaces(independent, known: [knownFace]).isEmpty)
    assert(!contextSupportsAdditionalFace(mouthFragment, known: [knownFace], evidence: [knownFace]))
    assert(!contextSupportsAdditionalFace(mouthFragment, known: [knownFace], evidence: []))
    assert(contextSupportsAdditionalFace(mouthFragment, known: [knownFace], evidence: [knownFace, mouthFragment]))
    assert(areaVolume(0, low: 1, high: 3, minimum: 0.1, maximum: 0.5) == 0.1)
    assert(abs(areaVolume(2, low: 1, high: 3, minimum: 0.1, maximum: 0.5) - 0.3) < 0.00001)
    assert(areaVolume(5, low: 1, high: 3, minimum: 0.1, maximum: 0.5) == 0.5)
    assert(shouldHideWindow(visible: true, miniaturized: false, active: true))
    assert(!shouldHideWindow(visible: false, miniaturized: false, active: true))
    assert(!shouldHideWindow(visible: true, miniaturized: true, active: true))
    assert(!shouldHideWindow(visible: true, miniaturized: false, active: false))
    assert(!needsAlarm(widths: []))
    assert(!needsAlarm(widths: [0.5]))
    assert(!needsAlarm(widths: [0.30]))
    assert(!needsAlarm(widths: [0.15]))
    assert(needsAlarm(widths: [0.5, 0.12]))
    assert(needsAlarm(widths: [0.5, 0.5]))
    assert(needsAlarm(widths: [0.01, 0.02, 0.03]))
    assert(!Detection(faces: []).shouldAlarm)
    assert(!Detection(faces: [CGRect(x: 0.2, y: 0.2, width: 0.02, height: 0.02)]).shouldAlarm)
    let tiles = detectionTiles()
    assert(tiles.count == 34)
    assert(Set((0..<34).map { ($0 * 13) % 34 }).count == 34)
    for point in [CGPoint(x: 0.01, y: 0.01), CGPoint(x: 0.99, y: 0.99), CGPoint(x: 0.9, y: 0.23), CGPoint(x: 0.5, y: 0.5)] {
        assert(tiles.contains { $0.contains(point) })
    }
    let a = CGRect(x: 0.1, y: 0.2, width: 0.1, height: 0.15)
    assert(mergeFaces([a, a.offsetBy(dx: 0.005, dy: 0.005)]).count == 1)
    assert(mergeFaces([a, a.offsetBy(dx: 0.2, dy: 0)]).count == 2)
    let near = CGRect(x: 0.1, y: 0.1, width: 0.4, height: 0.5)
    let distant = CGRect(x: 0.32, y: 0.2, width: 0.04, height: 0.05)
    assert(mergeFaces([near, distant]).count == 2)
    assert(mergeFaces([distant, near]).count == 2)
    // A shifted taller box from an open-mouth expression should not add a person.
    let yawning = CGRect(x: 0.095, y: 0.14, width: 0.11, height: 0.22)
    assert(mergeFaces([a, yawning]).count == 1)
    assert(mergeFaces([yawning, a]).count == 1)
    // Regression: the supplied screenshot has a face and a second mouth/chin box.
    let wholeFace = CGRect(x: 0.41, y: 0.27, width: 0.22, height: 0.44)
    let mouth = CGRect(x: 0.47, y: 0.26, width: 0.095, height: 0.19)
    let lowerTile = CGRect(x: 0.275, y: 0, width: 0.45, height: 0.45)
    assert(isCroppedFaceFragment(mouth, tile: lowerTile, fullFaces: [wholeFace]))
    let otherPerson = CGRect(x: 0.88, y: 0.13, width: 0.035, height: 0.06)
    assert(!isCroppedFaceFragment(otherPerson, tile: CGRect(x: 0.55, y: 0, width: 0.45, height: 0.45), fullFaces: [wholeFace]))
    assert(!isCroppedFaceFragment(mouth, tile: CGRect(x: 0, y: 0, width: 1, height: 1), fullFaces: [wholeFace]))
    assert(!isCroppedFaceFragment(mouth, tile: lowerTile, fullFaces: []))
    let display = previewRect(CGSize(width: 1920, height: 1080), in: CGRect(x: 0, y: 0, width: 600, height: 400))
    assert(abs(display.minY - 31.25) < 0.001)
    let topFace = fullFrameBox(CGRect(x: 0.1, y: 0.8, width: 0.1, height: 0.1), tile: display)
    let bottomFace = fullFrameBox(CGRect(x: 0.1, y: 0.1, width: 0.1, height: 0.1), tile: display)
    assert(topFace.minY > bottomFace.maxY)
    let mapped = fullFrameBox(CGRect(x: 0.5, y: 0.5, width: 0.1, height: 0.1), tile: CGRect(x: 0.4, y: 0.4, width: 0.6, height: 0.6))
    assert(abs(mapped.minX - 0.7) < 0.0001 && abs(mapped.width - 0.06) < 0.0001)
    print("PASS: 7 face-count cases + crop mapping + duplicate / distinct face cases")
} else {
    let app = NSApplication.shared
    let delegate = App()
    app.delegate = delegate
    app.run()
}
