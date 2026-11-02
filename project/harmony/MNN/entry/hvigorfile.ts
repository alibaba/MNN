import { hapTasks } from '@ohos/hvigor-ohos-plugin';
import {tcpkg_dependencies} from '@ali/tcpkg'

export default {
    system: hapTasks,  /* Built-in plugin of Hvigor. It cannot be modified. */
    plugins:[tcpkg_dependencies(),]         /* Custom plugin to extend the functionality of Hvigor. */
}
