import { harTasks } from '@ohos/hvigor-ohos-plugin';
import {mtl_plugin, tcpkg_dependencies, tcpkg_publish} from "@ali/tcpkg";

export default {
    system: harTasks,  /* Built-in plugin of Hvigor. It cannot be modified. */
    plugins:[mtl_plugin(), tcpkg_dependencies()]         /* Custom plugin to extend the functionality of Hvigor. */
}
