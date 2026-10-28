// RecaMem desktop shell (Tauri 2).
//
// The shell owns the window, the menu and the tray. All model behaviour lives in
// the Python API (`recamem serve --rest --port 8000`), which the app talks to
// over loopback; the shell never loads weights itself.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use tauri::menu::{Menu, MenuItem};
use tauri::tray::TrayIconBuilder;
use tauri::Manager;

fn main() {
    tauri::Builder::default()
        .setup(|app| {
            let refresh = MenuItem::with_id(app, "refresh", "Refresh", true, None::<&str>)?;
            let quit = MenuItem::with_id(app, "quit", "Quit", true, None::<&str>)?;
            let menu = Menu::with_items(app, &[&refresh, &quit])?;
            app.set_menu(menu)?;
            app.on_menu_event(|app, event| match event.id().as_ref() {
                "refresh" => {
                    if let Some(window) = app.get_webview_window("main") {
                        let _ = window.eval("window.location.reload()");
                    }
                }
                "quit" => app.exit(0),
                _ => {}
            });

            // Tray icon: shows and focuses the window instead of quitting.
            TrayIconBuilder::new()
                .tooltip("RecaMem")
                .on_tray_icon_event(|tray, _event| {
                    if let Some(window) = tray.app_handle().get_webview_window("main") {
                        let _ = window.show();
                        let _ = window.set_focus();
                    }
                })
                .build(app)?;
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running the RecaMem desktop shell");
}
