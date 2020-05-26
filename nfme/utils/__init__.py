def loading_bar(current_step, n_steps, prefix='', suffix='', bar_size=42):
    print(f"{prefix}{'x' * int(current_step / n_steps * bar_size)}" + \
          f"{'.' * int((1 - current_step / n_steps) * bar_size)}{suffix}",
          end='\r')
